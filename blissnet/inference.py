import numpy as np
import torch
import torch.nn as nn
from blissnet.models.blissnet import BLISSNetStage2

def relative_error(pred, true):
    pred = pred.ravel()
    true = true.ravel()
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))

def make_domain_coords(lats, lons):
    H, W = len(lats), len(lons)
    lat_n = (lats - lats.min()) / (lats.max() - lats.min() + 1e-9)
    lon_n = (lons - lons.min()) / (lons.max() - lons.min() + 1e-9)
    lg, logng = np.meshgrid(lat_n, lon_n, indexing='ij')
    coords = np.stack([lg.ravel(), logng.ravel()], axis=-1).astype(np.float32)
    return coords, H, W

def superres_coords(lats, lons, factor):
    H_new = len(lats) * factor
    W_new = len(lons) * factor
    lat_new = np.linspace(lats.min(), lats.max(), H_new)
    lon_new = np.linspace(lons.min(), lons.max(), W_new)
    return make_domain_coords(lat_new, lon_new)

class BLISSNetInference:
    def __init__(self, model, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        self.device = device
        self.model = model.to(device).eval()
        self._cached_basis = None
        self._cached_coords = None

    @classmethod
    def from_checkpoint(cls, ckpt_path, model_kwargs=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        saved = ckpt.get('args', {})
        K = ckpt.get('K', 512)

        defaults = dict(
            in_channels=saved.get('in_channels', 1),
            K=K,
            d_model=saved.get('d_model', 512),
            siren_width=saved.get('siren_width', 512),
            encoder_depth=saved.get('encoder_depth', 8),
            fg_size=saved.get('fg_size', 128),
            num_dec_blocks=saved.get('num_tf_blocks', 4),
            num_heads=saved.get('num_heads', 8),
        )
        
        if model_kwargs:
            defaults.update(model_kwargs)

        model = BLISSNetStage2(**defaults)
        model.load_state_dict(ckpt['model_state'])
        print(f"[BLISSNetInference] Loaded {ckpt_path} (K={K})")
        return cls(model, device=device)

    def cache_trunk(self, domain_coords):
        coords = domain_coords.to(self.device)
        self._cached_basis = self.model.precompute_trunk(coords)
        self._cached_coords = coords
        print(f"[BLISSNetInference] Trunk cached: {self._cached_basis.shape}")

    def clear_cache(self):
        self._cached_basis = None
        self._cached_coords = None

    @torch.no_grad()
    def predict(self, obs_values, obs_coords, domain_coords, use_cache=True):
        squeeze = obs_values.dim() == 2
        if squeeze:
            obs_values = obs_values.unsqueeze(0)
            obs_coords = obs_coords.unsqueeze(0)

        obs_values = obs_values.to(self.device)
        obs_coords = obs_coords.to(self.device)
        domain_coords = domain_coords.to(self.device)

        basis = None
        if use_cache and self._cached_basis is not None:
            if self._cached_coords is not None and \
               domain_coords.shape == self._cached_coords.shape and \
               torch.allclose(domain_coords, self._cached_coords, atol=1e-6):
                basis = self._cached_basis

        pred, _, _, _ = self.model(obs_values, obs_coords, domain_coords, precomputed_basis=basis)
        
        if squeeze:
            pred = pred.squeeze(0)
            
        return pred

    @torch.no_grad()
    def predict_numpy(self, obs_values, obs_coords, domain_coords, **kwargs):
        ov = torch.from_numpy(obs_values.astype(np.float32))
        oc = torch.from_numpy(obs_coords.astype(np.float32))
        dc = torch.from_numpy(domain_coords.astype(np.float32))
        out = self.predict(ov, oc, dc, **kwargs)
        return out.cpu().numpy()

    @staticmethod
    def bicubic_interpolate(obs_values, obs_coords, domain_coords, H, W, fill_value=None):
        from scipy.interpolate import griddata
        
        if fill_value is None:
            fill_value = float(np.nanmean(obs_values))
            
        pred = griddata(obs_coords, obs_values, domain_coords, method='cubic', fill_value=fill_value)
        return pred.reshape(H, W)

    @staticmethod
    def rbf_interpolate(obs_values, obs_coords, domain_coords, H, W, kernel='thin_plate_spline', smoothing=0.1):
        from scipy.interpolate import RBFInterpolator
        
        rbf = RBFInterpolator(obs_coords, obs_values[:, None], kernel=kernel, smoothing=smoothing)
        pred = rbf(domain_coords).squeeze()
        return pred.reshape(H, W)

    def evaluate(self, samples, domain_coords, H, W, run_baselines=True, verbose=True):
        dc_t = torch.from_numpy(domain_coords.astype(np.float32))
        self.cache_trunk(dc_t)

        blissnet_errs, bicubic_errs, rbf_errs = [], [], []

        for i, sample in enumerate(samples):
            ov = sample['obs_values'].astype(np.float32)
            oc = sample['obs_coords'].astype(np.float32)
            gt = sample['u_true_flat'].astype(np.float32)

            pred_np = self.predict_numpy(ov, oc, domain_coords)
            be = relative_error(pred_np, gt)
            blissnet_errs.append(be)

            if run_baselines:
                obs_1d = ov[:, 0] if ov.ndim == 2 else ov
                
                try:
                    bic = self.bicubic_interpolate(obs_1d, oc, domain_coords, H, W)
                    bicubic_errs.append(relative_error(bic.ravel(), gt))
                except Exception:
                    bicubic_errs.append(float('nan'))
                    
                try:
                    rbf = self.rbf_interpolate(obs_1d, oc, domain_coords, H, W)
                    rbf_errs.append(relative_error(rbf.ravel(), gt))
                except Exception:
                    rbf_errs.append(float('nan'))

            if verbose:
                line = f"  Sample {i:4d}  BLISSNet={be:.4f}"
                if run_baselines and bicubic_errs:
                    line += f"  Bicubic={bicubic_errs[-1]:.4f}"
                    line += f"  RBF={rbf_errs[-1]:.4f}"
                print(line)

        if verbose:
            be_arr = np.array(blissnet_errs)
            print(f"\n  BLISSNet — mean: {be_arr.mean():.4f} ± {be_arr.std():.4f}  median: {np.median(be_arr):.4f}")
            
            if run_baselines and bicubic_errs:
                bc = np.array([e for e in bicubic_errs if not np.isnan(e)])
                rb = np.array([e for e in rbf_errs if not np.isnan(e)])
                print(f"  Bicubic   — mean: {bc.mean():.4f} ± {bc.std():.4f}  median: {np.median(bc):.4f}")
                print(f"  RBF       — mean: {rb.mean():.4f} ± {rb.std():.4f}  median: {np.median(rb):.4f}")

        return {
            'blissnet': blissnet_errs,
            'bicubic': bicubic_errs,
            'rbf': rbf_errs
        }