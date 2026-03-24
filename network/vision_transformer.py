# network/vision_transformer.py
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ---------- Helpers ----------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ---------- PreNorm / FeedForward ----------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# ---------- Soma gating module ----------
class SomaTokenGate(nn.Module):
    """
    Per-token gating scores in (0,1). Optionally apply top-k mask (soft: -1e9).
    Input: token_feat [B, N, D]
    Output: gate [B, N]
    """
    def __init__(self, token_dim, use_projection=True, proj_hidden=None, top_k=None, temperature=1.0):
        super().__init__()
        self.use_projection = bool(use_projection)
        self.top_k = None if top_k is None else int(top_k)
        self.temperature = float(temperature)
        if self.use_projection:
            h = proj_hidden if proj_hidden is not None else max(16, token_dim // 8)
            self.proj = nn.Sequential(
                nn.Linear(token_dim, h),
                nn.GELU(),
                nn.Linear(h, 1)
            )
        else:
            self.query = nn.Parameter(torch.randn(token_dim) * 0.02)
        self.ln = nn.LayerNorm(token_dim)

    def forward(self, token_feat):
        # token_feat: [B, N, D]
        B, N, D = token_feat.shape
        H = self.ln(token_feat.reshape(-1, D)).reshape(B, N, D)
        if self.use_projection:
            scores = self.proj(H).squeeze(-1)  # [B,N]
        else:
            scores = torch.einsum('b n d, d -> b n', H, self.query)
        scores = scores / (self.temperature + 1e-12)

        if self.top_k is not None and 0 < self.top_k < N:
            topk_vals, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)  # [B,K]
            mask = torch.full_like(scores, float("-1e9"))
            mask = mask.scatter(-1, topk_idx, topk_vals)
            scores = mask

        gate = torch.sigmoid(scores)  # [B,N]
        return gate

# ---------- Attention (with optional SOMA gating) ----------
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_soma=False,
                 soma_topk=None, soma_proj_query=True, soma_proj_hidden=None, soma_temperature=1.0,
                 is_LSA=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.use_soma = bool(use_soma)
        if self.use_soma:
            token_dim = dim_head
            self.soma_gate = SomaTokenGate(token_dim,
                                           use_projection=soma_proj_query,
                                           proj_hidden=soma_proj_hidden,
                                           top_k=soma_topk,
                                           temperature=soma_temperature)

        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = None
        else:
            self.mask = None

    def forward(self, x):
        # x: [B, N, D]
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(qkv[0], 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(qkv[1], 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(qkv[2], 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,h,N,N]

        if self.use_soma:
            # gating computed from q mean across heads
            q_mean = q.mean(dim=1)  # [B,N,dim_head]
            gate = self.soma_gate(q_mean)  # [B,N]
            g_i = gate.unsqueeze(1).unsqueeze(-1)  # [B,1,N,1]
            g_j = gate.unsqueeze(1).unsqueeze(-2)  # [B,1,1,N]
            gating = g_i * g_j  # [B,1,N,N]
            gating = gating.expand(-1, self.heads, -1, -1)
            dots = dots * gating

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---------- Transformer stack ----------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 use_soma=False, soma_topk=None, soma_proj_query=True, soma_proj_hidden=None, soma_temperature=1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                      use_soma=use_soma, soma_topk=soma_topk,
                                      soma_proj_query=soma_proj_query, soma_proj_hidden=soma_proj_hidden,
                                      soma_temperature=soma_temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ---------- Cross-scale attention (supporting soma gating) ----------
class CrossScaleAttention(nn.Module):
    """
    CrossScaleAttention:
    - Simple self-attention on concatenated tokens from multiple scales.
    - Supports soma gating similar to Attention.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_soma=False,
                 soma_topk=None, soma_proj_query=True, soma_proj_hidden=None, soma_temperature=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.use_soma = bool(use_soma)
        if self.use_soma:
            self.soma_gate = SomaTokenGate(token_dim=dim_head,
                                           use_projection=soma_proj_query,
                                           proj_hidden=soma_proj_hidden,
                                           top_k=soma_topk,
                                           temperature=soma_temperature)

    def forward(self, x):
        # x: [B, N, D] where N is sum of tokens from all scales (+ cls)
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(qkv[0], 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(qkv[1], 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(qkv[2], 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.use_soma:
            q_mean = q.mean(dim=1)  # [B,N,dim_head]
            gate = self.soma_gate(q_mean)  # [B,N]
            g_i = gate.unsqueeze(1).unsqueeze(-1)
            g_j = gate.unsqueeze(1).unsqueeze(-2)
            gating = g_i * g_j
            gating = gating.expand(-1, self.heads, -1, -1)
            dots = dots * gating

        attn = torch.softmax(dots, dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---------- MultiScaleTransformer (NaViT-style multicropping) ----------
class MultiScaleTransformer(nn.Module):
    """
    Minimal NaViT-style multiscale:
      - Input: tokens_dict {'high','mid','low'} each [B, N, D]
      - Apply shared local transformer to each scale (weights can be shared)
      - Concatenate tokens, add a cls token, apply cross-scale attention (with soma optionally)
      - Return pooled feature vector [B, D]
    """
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.,
                 use_soma=False,
                 soma_topk=None,
                 soma_proj_query=True,
                 soma_proj_hidden=None,
                 soma_temperature=1.0,
                 pool='cls'):
        super().__init__()
        self.dim = dim
        self.pool = pool

        # local shared transformer (applied to each scale separately)
        self.shared_transform = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_soma=use_soma,
            soma_topk=soma_topk,
            soma_proj_query=soma_proj_query,
            soma_proj_hidden=soma_proj_hidden,
            soma_temperature=soma_temperature
        )

        # cross-scale fusion attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cross_attn = CrossScaleAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            use_soma=use_soma,
            soma_topk=soma_topk,
            soma_proj_query=soma_proj_query,
            soma_proj_hidden=soma_proj_hidden,
            soma_temperature=soma_temperature
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens_dict, H=None, W=None):
        # tokens_dict: {'high': [B,N1,D], 'mid': [B,N2,D], 'low': [B,N3,D]}
        th = self.shared_transform(tokens_dict['high'])
        tm = self.shared_transform(tokens_dict['mid'])
        tl = self.shared_transform(tokens_dict['low'])

        fused = torch.cat([th, tm, tl], dim=1)  # [B, Ntot, D]
        B = fused.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, fused), dim=1)  # [B, Ntot+1, D]
        x = self.dropout(x)
        x = self.cross_attn(x)  # [B, Ntot+1, D]

        out = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return out  # [B, D]

# ---------- Unified ViT (single-scale, compatible) ----------
class ViT(nn.Module):
    """
    Unified ViT:
      - Accepts img_size or image_size
      - mlp_dim or mlp_dim_ratio
      - to_patch_embedding should be set externally (SPT)
      - forward(tokens, H=None, W=None) supports pos interpolation (for single-scale tokens)
    """
    def __init__(self,
                 img_size=None,
                 image_size=None,
                 patch_size=16,
                 num_classes=1000,
                 dim=512,
                 depth=6,
                 heads=8,
                 mlp_dim=None,
                 mlp_dim_ratio=None,
                 pool='cls',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 use_soma=False,
                 soma_topk=None,
                 soma_proj_query=True,
                 soma_proj_hidden=None,
                 soma_temperature=1.0,
                 mlp_head='original'):
        super().__init__()

        resolved_img_size = img_size if img_size is not None else image_size
        if resolved_img_size is None:
            raise ValueError("ViT: one of img_size or image_size must be provided")
        image_height, image_width = pair(resolved_img_size)
        self.patch_size = patch_size

        # reference number of patches for base pos embedding
        self.ref_num_patches = (image_height // patch_size) * (image_width // patch_size)
        self.dim = dim
        self.num_classes = num_classes

        # mlp dim derivation
        if mlp_dim is None:
            if mlp_dim_ratio is None:
                mlp_dim = int(dim * 4)
            else:
                mlp_dim = int(dim * mlp_dim_ratio)
        self.mlp_dim = mlp_dim

        self.to_patch_embedding = None
        self.pos_embedding = nn.Parameter(torch.randn(1, self.ref_num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim=dim,
                                       depth=depth,
                                       heads=heads,
                                       dim_head=dim_head,
                                       mlp_dim=self.mlp_dim,
                                       dropout=dropout,
                                       use_soma=use_soma,
                                       soma_topk=soma_topk,
                                       soma_proj_query=soma_proj_query,
                                       soma_proj_hidden=soma_proj_hidden,
                                       soma_temperature=soma_temperature)
        self.pool = pool

        if mlp_head == 'original':
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        else:
            self.mlp_head = mlp_head

        self.image_size = (image_height, image_width)
        self.channels = channels
        self.dim_head = dim_head

    def interpolate_pos_encoding(self, x, H, W):
        """
        Safe interpolation from reference pos_embedding to new (H,W) grid.
        x: [B, N+1, D] including cls
        """
        B, N_plus_1, D = x.shape
        cls_pos = self.pos_embedding[:, :1]  # [1,1,D]
        patch_pos = self.pos_embedding[:, 1:]  # [1, N0, D]

        N0 = patch_pos.shape[1]
        new_N = N_plus_1 - 1
        if new_N == N0:
            return self.pos_embedding

        orig_size = int(math.sqrt(N0))
        if orig_size * orig_size != N0:
            orig_size = int(self.image_size[0] // self.patch_size)
            assert orig_size * orig_size == N0, f"pos_embed patch count {N0} not square and fallback failed."

        new_h = H // self.patch_size
        new_w = W // self.patch_size

        patch_pos = patch_pos.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, tokens, H=None, W=None):
        """
        tokens: [B, N, D] — produced by external SPT
        H, W: original image size used to compute token grid (for pos interpolation)
        """
        B, N, D = tokens.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, tokens), dim=1)

        if H is not None and W is not None:
            pos_embed = self.interpolate_pos_encoding(x, H, W)
        else:
            pos_embed = self.pos_embedding

        x = x + pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)
