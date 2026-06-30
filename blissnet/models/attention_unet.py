import torch
import torch.nn as nn
import torch.nn.functional as F

from blissnet.models.transformer_blocks import TransformerEncoderBlock, CoefficientDecoder

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        return x * self.psi(F.relu(g1 + x1, inplace=True))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attn = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()
        b = base_ch
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)
        self.out_conv = nn.Conv2d(b, 512, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out_conv(d1)

class Stage1Branch(nn.Module):
    def __init__(self, in_channels=1, K=512, d_model=512, num_tf_blocks=4, num_heads=8, base_ch=64):
        super().__init__()
        self.unet = AttentionUNet(in_channels=in_channels, base_ch=base_ch)
        self.linear_features = nn.Linear(512, d_model)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads)
            for _ in range(num_tf_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.to_coef = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K),
        )

    def forward(self, x):
        out = self.unet(x)
        out = self.pool(out)
        
        b, c, h, w = out.shape
        out = out.view(b, c, h * w).transpose(1, 2)
        
        out = self.linear_features(out)
        for blk in self.transformer_blocks:
            out = blk(out)
        out = self.norm(out)
        emb = out
        coefs = self.to_coef(out.mean(dim=1))
        
        return coefs, emb