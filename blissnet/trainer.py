import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from blissnet.models.blissnet import (
    BLISSNetStage1, BLISSNetStage2, Stage1Loss, BLISSNetLoss,
)

def build_sensor_index(sensor_idx_raw, batch_size, max_sensors, device):
    if isinstance(sensor_idx_raw, torch.Tensor):
        sensors = sensor_idx_raw.to(device)
        if sensors.shape[1] < max_sensors:
            pad_size = max_sensors - sensors.shape[1]
            padding = torch.zeros(batch_size, pad_size, dtype=torch.long, device=device)
            sensors = torch.cat([sensors, padding], dim=1)
        return sensors[:, :max_sensors]

    padded_sensors = torch.zeros(batch_size, max_sensors, dtype=torch.long, device=device)
    for i, seq in enumerate(sensor_idx_raw):
        length = min(len(seq), max_sensors)
        padded_sensors[i, :length] = seq[:length].to(device)
    return padded_sensors

class BLISSNetTrainer:
    def __init__(self, in_channels=1, K=512, d_model=512, siren_width=512, 
                 num_tf_blocks=4, num_heads=8, base_ch=64,
                 encoder_depth=8, fg_size=128,
                 epochs=60, lr=2e-5, grad_clip=1.0, lr_patience=5, lr_factor=0.5,
                 lambda_cp=10.0, lambda_coef=40.0, lambda_emb=0.01, lambda_gt=0.05):
        
        self.in_channels = in_channels
        self.K = K
        self.d_model = d_model
        self.siren_width = siren_width
        self.epochs = epochs
        self.grad_clip = grad_clip
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.s1_model = BLISSNetStage1(
            in_channels=in_channels, K=K, d_model=d_model, 
            siren_width=siren_width, num_tf_blocks=num_tf_blocks, 
            num_heads=num_heads, base_ch=base_ch
        ).to(self.device)

        self.s1_loss_fn = Stage1Loss()
        self.s1_optimizer = torch.optim.Adam(self.s1_model.parameters(), lr=lr)
        self.s1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.s1_optimizer, mode='min', factor=lr_factor, patience=lr_patience
        )
        self.s1_best_val_loss = float('inf')

        self.s2_model = BLISSNetStage2(
            in_channels=in_channels, K=K, d_model=d_model, 
            siren_width=siren_width, encoder_depth=encoder_depth, 
            num_heads=num_heads, fg_size=fg_size, num_dec_blocks=num_tf_blocks
        ).to(self.device)

        self.s2_loss_fn = BLISSNetLoss(
            lambda_cp=lambda_cp, lambda_coef=lambda_coef,
            lambda_emb=lambda_emb, lambda_gt=lambda_gt
        )
        
        self.s2_optimizer = None 
        self.s2_scheduler = None
        self.s2_best_val_loss = float('inf')
        self.is_stage2_ready = False

    def fit_stage1(self, train_loader, val_loader, domain_coords, checkpoint_path=None):
        print("--- Starting Stage 1 Training ---")
        coords = domain_coords.to(self.device)
        
        s1_metrics = {'train': [], 'val': []}

        for epoch in tqdm(range(1, self.epochs + 1)):
            start_time = time.time()

            train_loss = self._train_epoch_s1(train_loader, coords)
            val_loss = self._val_epoch_s1(val_loader, coords)
            
            self.s1_scheduler.step(val_loss)
            
            s1_metrics['train'].append(train_loss)
            s1_metrics['val'].append(val_loss)

            if val_loss < self.s1_best_val_loss:
                self.s1_best_val_loss = val_loss
                if checkpoint_path:
                    self._save_checkpoint(self.s1_model, checkpoint_path, epoch, val_loss)

            elapsed = time.time() - start_time
            print(f"Stage 1 | Epoch {epoch}/{self.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")
            
        return s1_metrics

    def _train_epoch_s1(self, loader, coords):
        self.s1_model.train()
        total_loss = 0.0
        
        for batch in loader:
            full_field, u_true = batch['full_field'].to(self.device), batch['u_true_flat'].to(self.device)

            self.s1_optimizer.zero_grad()
            predictions, _, _, _ = self.s1_model(full_field, coords)
            loss = self.s1_loss_fn(predictions, u_true)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.s1_model.parameters(), self.grad_clip)
            self.s1_optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def _val_epoch_s1(self, loader, coords):
        self.s1_model.eval()
        total_loss = 0.0
        
        for batch in loader:
            full_field, u_true = batch['full_field'].to(self.device), batch['u_true_flat'].to(self.device)
            
            predictions, _, _, _ = self.s1_model(full_field, coords)
            loss = self.s1_loss_fn(predictions, u_true)
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def prepare_stage2(self, lr=2e-5, lr_factor=0.5, lr_patience=5, freeze_teacher=True):
        self.s2_model.load_stage1_weights(self.s1_model, freeze=freeze_teacher)
        
        self.s1_model.eval()
        for param in self.s1_model.parameters():
            param.requires_grad = False

        trainable_params = [p for p in self.s2_model.parameters() if p.requires_grad]
        self.s2_optimizer = torch.optim.Adam(trainable_params, lr=lr)
        self.s2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.s2_optimizer, mode='min', factor=lr_factor, patience=lr_patience
        )
        
        self.is_stage2_ready = True

    def fit_stage2(self, train_loader, val_loader, domain_coords, checkpoint_path=None):
        print("\n--- Starting Stage 2 Training ---")
        coords = domain_coords.to(self.device)
        
        s2_metrics = {
            'train_total': [], 'val_total': [],
            'loss_cp': [], 'loss_coef': [], 'loss_emb': [], 'loss_gt': []
        }

        for epoch in tqdm(range(1, self.epochs + 1)):
            start_time = time.time()

            train_loss, dict_losses = self._train_epoch_s2(train_loader, coords)
            val_loss = self._val_epoch_s2(val_loader, coords)
            
            self.s2_scheduler.step(val_loss)
            
            s2_metrics['train_total'].append(train_loss)
            s2_metrics['val_total'].append(val_loss)
            s2_metrics['loss_cp'].append(dict_losses['cp'])
            s2_metrics['loss_coef'].append(dict_losses['coef'])
            s2_metrics['loss_emb'].append(dict_losses['emb'])
            s2_metrics['loss_gt'].append(dict_losses['gt'])

            if val_loss < self.s2_best_val_loss:
                self.s2_best_val_loss = val_loss
                if checkpoint_path:
                    self._save_checkpoint(self.s2_model, checkpoint_path, epoch, val_loss)

            elapsed = time.time() - start_time
            print(
                f"Stage 2 | Epoch {epoch}/{self.epochs} | "
                f"Train Total: {train_loss:.4f} | Val Total: {val_loss:.4f} | "
                f"CP: {dict_losses['cp']:.4f} | Coef: {dict_losses['coef']:.4f} | "
                f"Emb: {dict_losses['emb']:.4f} | GT: {dict_losses['gt']:.4f} | {elapsed:.1f}s"
            )
            
        return s2_metrics

    def _train_epoch_s2(self, loader, coords):
        self.s2_model.train()
        self.s1_model.eval()
        
        total_loss = 0.0
        running_components = {'cp': 0.0, 'coef': 0.0, 'emb': 0.0, 'gt': 0.0}

        for batch in loader:
            full_field = batch['full_field'].to(self.device)
            obs_values = batch['obs_values'].to(self.device)
            obs_coords = batch['obs_coords'].to(self.device)
            u_true = batch['u_true_flat'].to(self.device)
            
            b_size, max_s = obs_values.shape[0], obs_values.shape[1]
            sensor_mask = build_sensor_index(batch['sensor_idx'], b_size, max_s, self.device)

            with torch.no_grad():
                _, t_coefs, _, t_emb = self.s1_model(full_field, coords)

            self.s2_optimizer.zero_grad()
            pred, s_coefs, _, s_emb = self.s2_model(obs_values, obs_coords, coords)

            loss_outputs = self.s2_loss_fn(
                pred, s_coefs, s_emb, t_coefs, t_emb, u_true, sensor_mask=sensor_mask
            )
    
            if isinstance(loss_outputs, tuple):
                loss = loss_outputs[0]
                
                if len(loss_outputs) == 2 and isinstance(loss_outputs[1], dict):
                    for k, v in loss_outputs[1].items():
                        k_lower = k.lower()
                        val = v.item() if isinstance(v, torch.Tensor) else float(v)
                        if 'cp' in k_lower: running_components['cp'] += val
                        elif 'coef' in k_lower: running_components['coef'] += val
                        elif 'emb' in k_lower: running_components['emb'] += val
                        elif 'gt' in k_lower: running_components['gt'] += val
                        
                elif len(loss_outputs) >= 5:
                    running_components['cp'] += loss_outputs[1].item() if isinstance(loss_outputs[1], torch.Tensor) else float(loss_outputs[1])
                    running_components['coef'] += loss_outputs[2].item() if isinstance(loss_outputs[2], torch.Tensor) else float(loss_outputs[2])
                    running_components['emb'] += loss_outputs[3].item() if isinstance(loss_outputs[3], torch.Tensor) else float(loss_outputs[3])
                    running_components['gt'] += loss_outputs[4].item() if isinstance(loss_outputs[4], torch.Tensor) else float(loss_outputs[4])
            else:
                loss = loss_outputs

            loss.backward()
            nn.utils.clip_grad_norm_([p for p in self.s2_model.parameters() if p.requires_grad], self.grad_clip)
            self.s2_optimizer.step()
            
            total_loss += loss.item()

        num_batches = len(loader)
        averaged_components = {k: v / num_batches for k, v in running_components.items()}
        
        return total_loss / num_batches, averaged_components

    @torch.no_grad()
    def _val_epoch_s2(self, loader, coords):
        self.s2_model.eval()
        total_loss = 0.0

        for batch in loader:
            full_field = batch['full_field'].to(self.device)
            obs_values = batch['obs_values'].to(self.device)
            obs_coords = batch['obs_coords'].to(self.device)
            u_true = batch['u_true_flat'].to(self.device)
            
            b_size, max_s = obs_values.shape[0], obs_values.shape[1]
            sensor_mask = build_sensor_index(batch['sensor_idx'], b_size, max_s, self.device)

            _, t_coefs, _, t_emb = self.s1_model(full_field, coords)
            pred, s_coefs, _, s_emb = self.s2_model(obs_values, obs_coords, coords)

            loss_outputs = self.s2_loss_fn(
                pred, s_coefs, s_emb, t_coefs, t_emb, u_true, sensor_mask=sensor_mask
            )
            
            loss = loss_outputs[0] if isinstance(loss_outputs, tuple) else loss_outputs
            total_loss += loss.item()

        return total_loss / len(loader)
    
    def _save_checkpoint(self, model, path, epoch, val_loss):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'val_loss': val_loss,
            'K': self.K
        }, path)