import torch
import torch.nn as nn
import torch.nn.functional as F

from blissnet.models.siren import SIRENTrunk
from blissnet.models.attention_unet import Stage1Branch
from blissnet.models.transformer_blocks import (
    OFormerEncoder, FixedGridCrossAttention, CoefficientDecoder,
)

def reconstruct_field(coefs, basis):
    if basis.dim() == 2:
        return torch.matmul(coefs, basis.transpose(0, 1))
    
    coefs = coefs.unsqueeze(1)
    basis = basis.transpose(1, 2)
    return torch.bmm(coefs, basis).squeeze(1)

class BLISSNetStage1(nn.Module):
    def __init__(self, in_channels=1, K=512, d_model=512, siren_width=512, num_tf_blocks=4, num_heads=8, base_ch=64):
        super().__init__()
        self.K = K
        self.trunk = SIRENTrunk(coord_dim=2, hidden_width=siren_width, K=K)
        
        self.branch = Stage1Branch(
            in_channels=in_channels,
            K=K,
            d_model=d_model,
            num_tf_blocks=num_tf_blocks,
            num_heads=num_heads,
            base_ch=base_ch,
        )

    def forward(self, full_field, domain_coords):
        coefs, emb = self.branch(full_field)
        basis = self.trunk(domain_coords)
        pred = reconstruct_field(coefs, basis)
        return pred, coefs, basis, emb

    @torch.no_grad()
    def precompute_trunk(self, domain_coords):
        return self.trunk(domain_coords)

class BLISSNetStage2(nn.Module):
    def __init__(self, in_channels=1, K=512, d_model=512, siren_width=512, encoder_depth=8, num_heads=8, fg_size=128, num_dec_blocks=4):
        super().__init__()
        self.K = K
        self.in_channels = in_channels

        self.trunk = SIRENTrunk(coord_dim=2, hidden_width=siren_width, K=K)

        self.encoder = OFormerEncoder(
            in_channels=in_channels + 2, 
            d_model=d_model,
            num_heads=num_heads,
            depth=encoder_depth,
        )
        
        self.ca_bliss = FixedGridCrossAttention(
            d_model=d_model, num_heads=num_heads, fg_size=fg_size,
        )
        
        self.coef_decoder = CoefficientDecoder(
            fg_size=fg_size, d_model=d_model, K=K,
            num_transformer_blocks=num_dec_blocks, num_heads=num_heads,
        )

    def load_stage1_weights(self, stage1_model, freeze=True):
        self.trunk.load_state_dict(stage1_model.trunk.state_dict())
        
        if freeze:
            for param in self.trunk.parameters():
                param.requires_grad = False
    
            for param in self.coef_decoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def precompute_trunk(self, domain_coords):
        return self.trunk(domain_coords)

    def forward(self, obs_values, obs_coords, domain_coords, precomputed_basis=None):
        tokens = torch.cat([obs_values, obs_coords], dim=-1)
        phi = self.encoder(tokens)
        ca_out = self.ca_bliss(phi)
        coefs, emb = self.coef_decoder(ca_out)

        if precomputed_basis is not None:
            basis = precomputed_basis
        else:
            is_frozen = not next(self.trunk.parameters()).requires_grad
            grad_context = torch.no_grad() if is_frozen else torch.enable_grad()
            
            with grad_context:
                basis = self.trunk(domain_coords)

        pred = reconstruct_field(coefs, basis)
        return pred, coefs, basis, emb

class Stage1Loss(nn.Module):
    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)

class BLISSNetLoss(nn.Module):
    def __init__(self, lambda_cp=10.0, lambda_coef=40.0, lambda_emb=0.01, lambda_gt=0.05):
        super().__init__()
        self.lam_cp = lambda_cp
        self.lam_coef = lambda_coef
        self.lam_emb = lambda_emb
        self.lam_gt = lambda_gt

    def forward(self, pred2, coefs2, emb2, coefs1, emb1, u_true_flat, sensor_mask=None):
        denominator = (u_true_flat ** 2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        l_gt = ((pred2 - u_true_flat) ** 2 / denominator).mean()

        l_coef = F.mse_loss(coefs2, coefs1.detach())

        if emb1.shape == emb2.shape:
            l_emb = F.mse_loss(emb2, emb1.detach())
        else:
            l_emb = F.mse_loss(emb2.mean(dim=1), emb1.mean(dim=1).detach())

        if sensor_mask is not None:
            if sensor_mask.dtype == torch.bool:
                l_cp = F.mse_loss(pred2[sensor_mask], u_true_flat[sensor_mask])
            else:
                indices = sensor_mask.long()
                student_sensor_preds = pred2.gather(1, indices)
                actual_sensor_truths = u_true_flat.gather(1, indices)
                l_cp = F.mse_loss(student_sensor_preds, actual_sensor_truths)
        else:
            l_cp = l_gt   

        total_loss = (self.lam_cp * l_cp) + \
                     (self.lam_coef * l_coef) + \
                     (self.lam_emb * l_emb) + \
                     (self.lam_gt * l_gt)
                     
        loss_breakdown = {
            'l_cp': l_cp.item(),
            'l_coef': l_coef.item(),
            'l_emb': l_emb.item(),
            'l_gt': l_gt.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_breakdown