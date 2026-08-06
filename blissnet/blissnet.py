import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class MHA(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1, two_dim=False):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.weights = nn.Linear(emb_dim, emb_dim * 3)
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.two_dim = two_dim

    def forward(self, x):
        if self.two_dim:
            B, emb_dim, height, width = x.shape
            x = x.reshape(B, emb_dim, height*width).permute(0, 2, 1)

        B, S, emb_dim = x.shape
        H = self.num_heads
        head_size = int(emb_dim / H)

        qkv = self.weights(x).reshape(B, S, H, 3, head_size)
        Q, K, V = qkv.unbind(dim=-2)          # B, S, H, head_size
        Q, K, V = (t.permute(0, 2, 1, 3) for t in (Q, K, V))   # B, H, S, head_size

        output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p if self.training else 0.0
        )  # B, H, S, head_size

        output = output.permute(0, 2, 1, 3).reshape(B, S, emb_dim)
        output = self.projection(output)
        if self.two_dim:
            output = output.permute(0, 2, 1).reshape(B, emb_dim, height, width)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, dropout=0.1):
        super().__init__()

        self.gelu1 = nn.GELU()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.gelu2 = nn.GELU()
        self.gn2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.is_projection = False
        if in_channels != out_channels:
            self.is_projection = True
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x.clone()

        out = self.gelu1(x)
        out = self.gn1(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.gelu2(out)
        out = self.gn2(out)
        out = self.conv2(out)

        if self.is_projection:
            res = self.projection(res)

        out = out + res
        return out



class AttentionUNet(nn.Module):
    def __init__(self, in_channels, base_channels, n_heads, n_groups, dropout=0.1):
        super().__init__()

        c1, c2, c3, c4 = base_channels, base_channels*2, base_channels*4, base_channels*8

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, c1),
            nn.GELU(),
            nn.MaxPool2d(2)
            
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, c2),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(n_groups, c3),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        
        self.bn = nn.Sequential(
            ResidualBlock(c3, c4, n_groups, dropout),
            nn.GroupNorm(1, c4),
            MHA(c4, n_heads, dropout, two_dim=True),
            ResidualBlock(c4, c3, n_groups, dropout)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2),
            nn.GroupNorm(n_groups, c2),
            nn.GELU()
        )

        self.conv2 = nn.Conv2d(c2*2, c2, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2),
            nn.GroupNorm(n_groups, c1),
            nn.GELU()
        )

        self.conv3 = nn.Conv2d(c1*2, c1, kernel_size=3, stride=1, padding=1)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c1, in_channels, kernel_size=2, stride=2),
        )

        self.conv4 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        res1 = x.clone()

        res2 = self.enc1(res1)
        res3 = self.enc2(res2)
        out_enc = self.enc3(res3)

        bn_out = self.bn(out_enc)

        out1 = self.dec3(bn_out)
        if out1.shape[-2:] != res3.shape[-2:]:
            out1 = F.interpolate(out1, size=res3.shape[-2:], mode='nearest')

        out2 = torch.cat([out1, res3], dim=1)
        out2 = self.conv2(out2)
        out2 = self.dec2(out2)

        if out2.shape[-2:] != res2.shape[-2:]:
            out2 = F.interpolate(out2, size=res2.shape[-2:], mode='nearest')
        out3 = torch.cat([out2, res2], dim=1)
        out3 = self.conv3(out3)
        out3 = self.dec1(out3)

        if out3.shape[-2:] != res1.shape[-2:]:
            out3 = F.interpolate(out3, size=res1.shape[-2:], mode='nearest')
        out4 = torch.cat([out3, res1], dim=1)
        out4 = self.conv4(out4)

        return out4


class FFN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.w1 = nn.Linear(in_dim, in_dim*4)
        self.w2 = nn.Linear(in_dim, in_dim*4)
        self.silu = nn.SiLU()
        self.w3 = nn.Linear(in_dim*4, in_dim)

    def forward(self, x):
        out1 = self.silu(self.w1(x))
        out2 = self.w2(x)
        out = out1 * out2
        out = self.w3(out)

        return out

class Transformer(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout=0.1):
        super().__init__()

        self.attention_block = nn.Sequential(
            nn.LayerNorm(emb_dim),
            MHA(emb_dim, n_heads, dropout, two_dim=False)
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            FFN(emb_dim)
        )

    def forward(self, x):
        res1 = x.clone()

        out1 = self.attention_block(x)
        out1 = out1 + res1
        res2 = out1.clone()
        out2 = self.ffn(out1)
        out2 = out2 + res2
        return out2

class SineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=1, first_layer=False):
        super().__init__()

        self.omega = omega
        self.fn = nn.Linear(in_dim, out_dim)

        self.init_weights(first_layer)

    @torch.no_grad()
    def init_weights(self, is_first):
        if is_first:
            bound = 1 / self.fn.in_features
        else:
            bound = math.sqrt(6 / self.fn.in_features) / self.omega
        
        self.fn.weight.uniform_(-bound, bound)
        if self.fn.bias is not None:
            self.fn.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.fn(x))

class SIREN(nn.Module):
    def __init__(self, hidden_dim, n_layers, out_K, omega=30):
        super().__init__()

        self.model = nn.Sequential(
            SineLayer(2, hidden_dim, omega, first_layer=True),
            *[SineLayer(hidden_dim, hidden_dim, 1) for _ in range(n_layers)],
            nn.Linear(hidden_dim, out_K)
        )

    def forward(self, x):
        return self.model(x)



class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout):
        super().__init__()

        self.value_encoder = nn.Linear(1, emb_dim)
        self.coord_encoder = nn.Linear(2, emb_dim)
        self.transformer = Transformer(emb_dim, n_heads, dropout)

    def forward(self, x, coord):
        x_emb = self.value_encoder(x)
        coord_emb = self.coord_encoder(coord)
        emb = x_emb + coord_emb
        output = self.transformer(emb)
        return output


class CrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.q_w = nn.Linear(emb_dim, emb_dim)
        self.kv_w = nn.Linear(emb_dim, emb_dim * 2)
        self.projection = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x, y):
        B, S_A, emb_dim = x.shape
        _, S_B, _ = y.shape

        H = self.num_heads
        head_size = int(emb_dim / H)

        Q = self.q_w(x).reshape(B, S_A, H, head_size).permute(0, 2, 1, 3)  # B,H,S_A,head_size  <-- fix

        kv = self.kv_w(y).reshape(B, S_B, H, 2, head_size)
        K, V = kv.unbind(dim=-2)
        K = K.permute(0, 2, 1, 3)   # B,H,S_B,head_size
        V = V.permute(0, 2, 1, 3)

        output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0
        )  # B,H,S_A,head_size

        output = output.permute(0, 2, 1, 3).reshape(B, S_A, emb_dim)
        output = self.projection(output)
        return output

class BranchNet1(nn.Module):
    def __init__(self, emb_dim, H, W, K, in_channels, base_channels, n_heads, n_groups, dropout=0.1, n_transformer_layers=4, n_hidden_linear_layers=3, pool_factor=4):
        super().__init__()

        self.K = K
        self.pool_factor = pool_factor

        self.att_unet = AttentionUNet(in_channels, base_channels, n_heads, n_groups, dropout)
        self.pool = nn.AvgPool2d(pool_factor)
        self.linear_proj = nn.Linear(in_channels, emb_dim)

        self.transformer_blocks = nn.Sequential(
            *[Transformer(emb_dim, n_heads, dropout) for _ in range(n_transformer_layers)]
        )
        
        S = (H // pool_factor) * (W // pool_factor) 
        
        exp_dim = emb_dim * 2 
        
        self.mlp_decoder = nn.Sequential(
            nn.Linear(emb_dim, 64),           
            nn.Flatten(start_dim=1),       
            nn.Linear(S * 64, exp_dim),           
            nn.SiLU(),
            *[nn.Sequential(nn.Linear(exp_dim, exp_dim), nn.SiLU()) for _ in range(n_hidden_linear_layers - 1)],
            nn.Linear(exp_dim, K)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        output = self.att_unet(x)
        output = self.pool(output)
        Hp, Wp = output.shape[-2:]                  
        output = output.reshape(B, C, Hp*Wp).permute(0, 2, 1)
        embedded_out = self.linear_proj(output) # B, S, emb_dim
        output = embedded_out
        for layer in self.transformer_blocks:
            output = checkpoint(layer, output, use_reentrant=False)
            
        coeffs = self.mlp_decoder(output) # Output shape: (B, K)
        return coeffs, embedded_out

class FourierFeatureTransform(nn.Module):
    def __init__(self, feature_size, sigma=1):
        super().__init__()

        self.B = nn.Parameter(torch.randn(2, feature_size // 2) * sigma)

    def forward(self, x):
        proj = 2 * torch.pi * (x @ self.B)
        out = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return out


class BranchNet2(nn.Module):
    def __init__(self, frozen_transformer:nn.Module, frozen_mlp:nn.Module, grid_resolution, emb_dim, n_heads, dropout=0.1, K=None):
        super().__init__()

        grid_tensor = self.generate_grid(grid_resolution) # res, 2
        self.register_buffer('fixed_grid', grid_tensor)

        self.fourier_feature_transform = FourierFeatureTransform(emb_dim, 1)

        self.transformer_encoder = TransformerEncoder(emb_dim, n_heads, dropout)
        self.cross_att = CrossAttention(emb_dim, n_heads, dropout)
        self.transformer_blocks = frozen_transformer
        self.mlp_decoder = frozen_mlp
        self.K = K

    def generate_grid(self, length):
        vec_a = torch.linspace(0, 1, length)
        vec_b = torch.linspace(0, 1, length)

        grid = torch.stack([vec_a, vec_b], dim=-1)
        return grid

    def forward(self, x, coord):
        input_emb = self.transformer_encoder(x, coord)

        fourier_map = self.fourier_feature_transform(self.fixed_grid)
        fourier_map = fourier_map.expand(input_emb.shape[0], -1, -1)

        emb_output = self.cross_att(fourier_map, input_emb)
        output = emb_output
        for layer in self.transformer_blocks:
            output = checkpoint(layer, output, use_reentrant=False)
            
        coeffs = self.mlp_decoder(output) # Output shape: (B, K)
        return coeffs, emb_output
    
class BLISSNet(nn.Module):
    def __init__(self, emb_dim, H, W, K, n_heads, dropout=0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.H = H
        self.W = W
        self.K = K
        self.n_heads = n_heads
        self.dropout = dropout
        self.phase = -1
        
    def setPhase(self, phase, config):
        if phase == self.phase:
            return
        
        self.phase = phase
        if phase == 0:
            self.branch1 = BranchNet1(self.emb_dim, self.H, self.W, self.K, config.in_channels, config.base_channels, self.n_heads, config.n_groups, self.dropout, config.n_transformer_layers, config.n_hidden_linear_layers, config.pool_factor)

            self.trunk_net = SIREN(config.siren_hidden_dim, config.siren_layers, self.K, config.omega)
        else:
            for param in self.trunk_net.parameters():
                param.requires_grad = False

            for param in self.branch1.parameters():
                param.requires_grad = False

            self.branch1.eval()
            self.trunk_net.eval()

            self.branch2 = BranchNet2(self.branch1.transformer_blocks, self.branch1.mlp_decoder, config.grid_size, self.emb_dim, self.n_heads, self.dropout, self.K)

    def generate_grid(self, height, width, device=None):
        vec_a = torch.linspace(0, 1, height).to(device)
        vec_b = torch.linspace(0, 1, width).to(device)

        grid_x, grid_y = torch.meshgrid(vec_a, vec_b, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)

        return grid
    
    def forward(self, x, resolution, phase=0):
        H, W = resolution
        device = x.device if phase == 0 else x[0].device

        grid = self.generate_grid(H, W, device).unsqueeze(0)
        basis = self.trunk_net(grid) # 1, H, W, K

        if phase == 0:
            coeff_map, embedded_output = self.branch1(x) # B, K
        else:
            x_vals, coords = x
            coeff_map, embedded_output = self.branch2(x_vals, coords) # B, K

        coeffs = coeff_map.view(-1, 1, 1, self.K) 
    
        output = (basis * coeffs).sum(dim=-1, keepdim=True)

        return output.permute(0, 3, 1, 2), coeff_map, embedded_output