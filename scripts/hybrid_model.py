# scripts/hybrid_model.py
"""
OceanEmbed Hybrid Model (CNN + ViT + Center Skip + Physics Loss)
Robust architecture for 7-variable ocean subsurface reconstruction:
- Center-preserving multi-scale CNN
- Lightweight ViT branch for spatial correlation
- Direct point-wise surface skip connection (SST, SSS, SSH, currents, winds)
- Gravitational vertical stability physics loss
"""
import torch
import torch.nn as nn


# ============================================================
# CNN BRANCH (Center-Preserving Spatial Context)
# ============================================================
class CNNBranch(nn.Module):
    def __init__(self, in_channels=7, out_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
        # Center crop linear projection + global pool projection
        self.fc_center = nn.Linear(out_dim, out_dim)
        self.fc_context = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: (B, C, H, W) where H=W=8
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)  # (B, out_dim, 8, 8)
        
        # Center core representation (2x2 center of the 8x8 patch)
        center_core = h[:, :, 3:5, 3:5].mean(dim=(2, 3))
        # Global context (meso-scale eddy / current divergence across patch)
        context = h.mean(dim=(2, 3))
        
        out = self.fc_center(center_core) + 0.5 * self.fc_context(context)
        return out


# ============================================================
# ViT BRANCH (Spatial Multi-Head Attention)
# ============================================================
class ViTBranch(nn.Module):
    def __init__(self, in_channels=7, patch_size=8, token_dim=64,
                 num_layers=1, num_heads=4, tokens_side=4):
        super().__init__()
        self.patch_size = patch_size
        self.tokens_side = tokens_side
        self.token_size = patch_size // tokens_side  # 2
        self.num_tokens = tokens_side ** 2          # 16

        token_input_dim = in_channels * self.token_size * self.token_size
        self.patch_embed = nn.Linear(token_input_dim, token_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=token_dim * 2, dropout=0.1,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, P, _ = x.shape
        ts = self.token_size
        n = self.tokens_side

        x = x.view(B, C, n, ts, n, ts)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_tokens, -1)

        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]  # CLS output


# ============================================================
# HYBRID MODEL (CNN + ViT + Center Skip)
# ============================================================
class OceanEmbedHybrid(nn.Module):
    def __init__(self, in_channels=7, patch_size=8, n_depths=15,
                 cnn_out=128, vit_dim=64, embed_dim=256):
        super().__init__()
        self.cnn = CNNBranch(in_channels, out_dim=cnn_out)
        self.vit = ViTBranch(in_channels, patch_size=patch_size, token_dim=vit_dim)
        
        # Center-pixel point-wise projection (direct surface-to-depth pathway)
        self.center_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )

        fused_dim = cnn_out + vit_dim + 64  # 128 + 64 + 64 = 256

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_depths)
        )

    def _extract_center(self, x):
        # 2x2 center average of patch: indices (3..4, 3..4)
        return x[:, :, 3:5, 3:5].mean(dim=(2, 3))

    def forward(self, x):
        c = self.cnn(x)
        v = self.vit(x)
        p = self.center_proj(self._extract_center(x))
        e = self.fusion(torch.cat([c, v, p], dim=1))
        return self.decoder(e)

    def get_embedding(self, x):
        c = self.cnn(x)
        v = self.vit(x)
        p = self.center_proj(self._extract_center(x))
        return self.fusion(torch.cat([c, v, p], dim=1))


# ============================================================
# PHYSICS LOSS (Gravitational Stability & Deep Regularization)
# ============================================================
def physics_loss(pred_temps, depths_tensor):
    """
    Enforces oceanographic physical constraints:
    1. Static thermal stability: Temperature must generally decrease with depth (dT/dz <= 0).
       Penalizes inverted water columns below the surface mixed layer (depth >= 50m).
    2. Deep ocean smoothness: Deep abyssal waters (>= 300m) have smooth gradients without oscillations.
    Crucially, DOES NOT flatten the sharp 50m-150m thermocline.
    """
    # dT between consecutive depth levels (T_{k+1} - T_k)
    dT = torch.diff(pred_temps, dim=1)  # shape: (B, 14)
    
    # 1. Thermal inversion penalty for depth levels >= 50m (index 5 to 14)
    # Allows tiny observational variations (up to 0.1 C), penalizes larger inversions
    inversions = torch.relu(dT[:, 5:] - 0.1)
    loss_stability = torch.mean(inversions ** 2)

    # 2. Deep stratification smoothness (depth >= 300m: indices 10 to 14)
    dz = torch.diff(depths_tensor).unsqueeze(0).clamp(min=1.0)
    first_deriv = dT / dz
    d2T_deep = torch.diff(first_deriv[:, 10:], dim=1)
    loss_smooth = torch.mean(d2T_deep ** 2)

    return loss_stability + 1e-4 * loss_smooth