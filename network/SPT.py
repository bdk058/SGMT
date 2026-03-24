import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

class ShiftedPatchTokenization(nn.Module):
    """
    Dynamic Shifted Patch Tokenization (SPT-V2)
    Allows variable-size image input without enforcing square shape.
    """
    def __init__(self, in_dim, dim, patch_size=4, exist_class_t=False):
        super().__init__()
        self.exist_class_t = exist_class_t
        self.patch_size = patch_size

        self.patch_shifting = PatchShifting(patch_size)

        # Each pixel is concatenated with its 4 diagonal shifts → 5× input channels
        patch_dim = (in_dim * 5) * (patch_size ** 2)

        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.merging = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        """
        Input: x [B, C, H, W]
        Output: tokens [B, N_patches, D]
        """
        B, C, H, W = x.shape

        # Shift-augmented representation
        x_shifted = self.patch_shifting(x)

        # Extract patches dynamically
        patch_size = self.patch_size
        H_p, W_p = H // patch_size, W // patch_size

        # Trim excess borders so patching divides evenly
        x_shifted = x_shifted[:, :, :H_p * patch_size, :W_p * patch_size]

        tokens = rearrange(
            x_shifted,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=patch_size,
            p2=patch_size
        )
        tokens = self.merging(tokens)

        # Handle optional CLS token
        if self.exist_class_t:
            cls_token = self.class_linear(torch.mean(x, dim=[2, 3], keepdim=False)).unsqueeze(1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        return tokens


class PatchShifting(nn.Module):
    """
    Spatial context expansion via 4-diagonal shifting.
    """
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * 0.5)

    def forward(self, x):
        pad = self.shift
        x_pad = F.pad(x, (pad, pad, pad, pad))

        x_lu = x_pad[:, :, :-2 * pad, :-2 * pad]
        x_ru = x_pad[:, :, :-2 * pad, 2 * pad:]
        x_lb = x_pad[:, :, 2 * pad:, :-2 * pad]
        x_rb = x_pad[:, :, 2 * pad:, 2 * pad:]

        # Concatenate central + 4 diagonals → richer context
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        return x_cat



class MultiScaleSPT(nn.Module):
    """
    Create tokens at three scales:
      - high:  base_patch
      - mid:   base_patch * 2
      - low:   base_patch * 4
    Implementation: instantiate three ShiftedPatchTokenization modules.
    Returns dict: {'high': tokens_h, 'mid': tokens_m, 'low': tokens_l}
    Each tokens_* shape: [B, N_scale, D]
    """
    def __init__(self, in_dim, dim, base_patch=4, exist_class_t=False):
        super().__init__()
        self.base_patch = int(base_patch)
        # three SPTs with different patch sizes
        self.spt_high = ShiftedPatchTokenization(in_dim, dim, patch_size=self.base_patch, exist_class_t=exist_class_t)
        self.spt_mid  = ShiftedPatchTokenization(in_dim, dim, patch_size=self.base_patch * 2, exist_class_t=exist_class_t)
        self.spt_low  = ShiftedPatchTokenization(in_dim, dim, patch_size=self.base_patch * 4, exist_class_t=exist_class_t)

    def forward(self, x):
        # x: [B, C, H, W]
        # Each SPT internally trims to make H,W divisible by its patch size.
        tokens_h = self.spt_high(x)  # [B, N_h, D]
        tokens_m = self.spt_mid(x)   # [B, N_m, D]
        tokens_l = self.spt_low(x)   # [B, N_l, D]
        return {'high': tokens_h, 'mid': tokens_m, 'low': tokens_l}
