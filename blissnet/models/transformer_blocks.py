import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatureEncoding(nn.Module):
    def __init__(self, coord_dim=2, d_model=512, num_freqs=64):
        super().__init__()
        self.register_buffer('B_mat', torch.randn(coord_dim, num_freqs) * 10.0)
        self.proj = nn.Linear(num_freqs * 2, d_model)

    def forward(self, coords):
        squeeze = coords.dim() == 2
        if squeeze:
            coords = coords.unsqueeze(0)
        x = torch.matmul(coords, self.B_mat)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.proj(x)
        return x.squeeze(0) if squeeze else x

class GalerkinAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_k = nn.LayerNorm(self.head_dim)
        self.norm_v = nn.LayerNorm(self.head_dim)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        
        k = self.norm_k(k)
        v = self.norm_v(v)
        
        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv) * self.scale
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.out_proj(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ffn_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GalerkinAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OFormerEncoder(nn.Module):
    def __init__(self, in_channels=3, d_model=512, num_heads=8, depth=8, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs_tokens):
        x = self.input_proj(obs_tokens)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class FixedGridCrossAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, fg_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.fg_size = fg_size

        t = torch.linspace(0.0, 1.0, fg_size)
        fg = torch.stack([t, t], dim=-1)
        self.register_buffer('fixed_grid', fg)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc = FourierFeatureEncoding(coord_dim=2, d_model=d_model)

    def forward(self, phi):
        b = phi.shape[0]
        n_k = phi.shape[1]
        
        fg = self.fixed_grid.unsqueeze(0).expand(b, -1, -1)
        q_pos = self.pos_enc(fg)
        n_q = q_pos.shape[1]

        q = self.q_proj(q_pos).view(b, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(phi).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(phi).view(b, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n_q, -1)
        return self.norm(self.out_proj(out))

class CoefficientDecoder(nn.Module):
    def __init__(self, fg_size=128, d_model=512, K=512, num_transformer_blocks=4, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads)
            for _ in range(num_transformer_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.to_coef = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        emb = x
        coefs = self.to_coef(x.mean(dim=1))
        return coefs, emb