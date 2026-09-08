#cell 1
from google.colab import drive
drive.mount('/content/drive')

import os

SAVE_DIR = "/content/drive/MyDrive/SGMT_cifar100_no_DNM_checkpoint"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("Checkpoint dir:", SAVE_DIR)

!git clone https://github.com/cyizhuo/CIFAR-100-dataset.git

DATA_DIR = "./CIFAR-100-dataset"

#cell 2
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

def dynamic_collate(batch):

    imgs, labels = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = []
    for img in imgs:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        img = F.pad(img,(pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        padded.append(img)
    return torch.stack(padded), torch.tensor(labels)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3,(0.5,) * 3),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3,(0.5,) * 3),
])

train_set = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_set = ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, collate_fn=dynamic_collate)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=dynamic_collate)

print(f"Train Dataset : {len(train_set)}")
print(f"Test Dataset  : {len(test_set)}")

#cell 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchShifting(nn.Module):
    def __init__(self, shift_ratio=0.5):
        super().__init__()
        self.shift_ratio = shift_ratio

    def forward(self, x, patch_size):
        shift = max(1, int(patch_size * self.shift_ratio))
        pad = shift
        x_pad = F.pad(x, (pad, pad, pad, pad))
        x_lu = x_pad[:, :, :-2 * pad, :-2 * pad]
        x_ru = x_pad[:, :, :-2 * pad, 2 * pad:]
        x_lb = x_pad[:, :, 2 * pad:, :-2 * pad]
        x_rb = x_pad[:, :, 2 * pad:, 2 * pad:]
        return torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, patch_size=4, exist_class_t=False, use_conv_stem=True):
        super().__init__()
        self.patch_size = patch_size
        self.exist_class_t = exist_class_t
        self.use_conv_stem = use_conv_stem
        if use_conv_stem:
            self.conv_stem = nn.Sequential(
                nn.Conv2d(in_dim, dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim // 2),
                nn.GELU()
            )
            stem_dim = dim // 2
            self.local_mixer = nn.Sequential(
                nn.Conv2d(
                    stem_dim * 5,
                    stem_dim * 5,
                    kernel_size=3,
                    padding=1,
                    groups=stem_dim * 5
                ),
                nn.BatchNorm2d(stem_dim * 5),
                nn.GELU()
            )
        else:
            self.conv_stem = nn.Identity()
            stem_dim = in_dim
            self.local_mixer = nn.Identity()

        self.patch_shift = PatchShifting()
        shifted_dim = stem_dim * 5
        patch_dim = shifted_dim * (patch_size ** 2)
                
        self.proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        if exist_class_t:
            self.cls_proj = nn.Linear(stem_dim, dim)
                
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_stem(x)
        x = self.patch_shift(x, self.patch_size)
        x = self.local_mixer(x)
        p = self.patch_size
        H_p = H // p
        W_p = W // p
        x = x[:, :, :H_p * p, :W_p * p]
        tokens = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.proj(tokens)

        if self.exist_class_t:
            cls = self.cls_proj(
                F.adaptive_avg_pool2d(
                    x[:, :x.shape[1] // 5],
                    1
                ).flatten(1)
            ).unsqueeze(1)
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens

class MultiScaleSPT(nn.Module):
    def __init__(self, in_dim, dim, base_patch=4, exist_class_t=False, use_conv_stem=True):
        super().__init__()
        self.high = ShiftedPatchTokenization(
            in_dim,
            dim,
            patch_size=base_patch,
            exist_class_t=exist_class_t,
            use_conv_stem=use_conv_stem
        )
        self.mid = ShiftedPatchTokenization(
            in_dim,
            dim,
            patch_size=base_patch * 2,
            exist_class_t=exist_class_t,
            use_conv_stem=use_conv_stem
        )
        self.low = ShiftedPatchTokenization(
            in_dim,
            dim,
            patch_size=base_patch * 4,
            exist_class_t=exist_class_t,
            use_conv_stem=use_conv_stem
        )
                
    def forward(self, x):
        return {
            "high": self.high(x),
            "mid": self.mid(x),
            "low": self.low(x)
        }

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

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
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_proj, gate = self.proj(x).chunk(2, dim=-1)
        x = x_proj * F.silu(gate)
        return self.out(x)

class SomaTokenGate(nn.Module):
    def __init__(self, token_dim, use_projection=True, proj_hidden=None, top_k=None, temperature=1.0):
        super().__init__()
        self.use_projection = use_projection
        self.top_k = top_k
        self.temperature = nn.Parameter(
            torch.tensor(float(temperature))
        )
        if self.use_projection:
            h = proj_hidden if proj_hidden else max(16, token_dim // 4)
            self.proj = nn.Sequential(
                nn.Linear(token_dim, h),
                nn.GELU(),
                nn.Linear(h, 1)
            )
        else:
            self.query = nn.Parameter(
                torch.randn(token_dim) * 0.02
            )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        x = self.norm(x)
        if self.use_projection:
            scores = self.proj(x).squeeze(-1)
        else:
            scores = torch.einsum('b n d, d -> b n', x, self.query)
        scores = scores / (self.temperature.abs() + 1e-6)

        if self.top_k is not None and self.top_k < scores.shape[-1]:
            topk_vals, topk_idx = torch.topk(
                scores,
                k=self.top_k,
                dim=-1
            )
            mask = torch.full_like(scores, -1e9)
            mask.scatter_(-1, topk_idx, topk_vals)
            scores = mask
        gate = torch.softmax(scores, dim=-1)
        return gate

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., use_soma=False, soma_topk=None, soma_proj_query=True,
                 soma_proj_hidden=None,
                 soma_temperature=1.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout)
        self.use_soma = use_soma

        if self.use_soma:
            self.soma_gate = SomaTokenGate(
                token_dim=dim_head,
                use_projection=soma_proj_query,
                proj_hidden=soma_proj_hidden,
                top_k=soma_topk,
                temperature=soma_temperature
            )

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(qkv[0], 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(qkv[1], 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(qkv[2], 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.einsum(
            'b h i d, b h j d -> b h i j',
            q,
            k
        ) * self.scale
        dots = dots - dots.max(dim=-1, keepdim=True)[0]
        attn = torch.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum(
            'b h i j, b h j d -> b h i d',
            attn,
            v
        )

        if self.use_soma:
            q_mean = q.mean(dim=1)
            gate = self.soma_gate(q_mean)
            gate = gate.unsqueeze(1).unsqueeze(-1)
            out = out * (1 + gate)
        out = rearrange(
            out,
            'b h n d -> b n (h d)'
        )
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout=0.,
                 drop_path=0.1,
                 use_soma=False,
                 soma_topk=None,
                 soma_proj_query=True,
                 soma_proj_hidden=None,
                 soma_temperature=1.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dp = drop_path * (i / max(depth - 1, 1))
            attn = PreNorm(
                dim,
                Attention(
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
            )
            ff = PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout))

            block = nn.ModuleDict({
                "attn": attn,
                "ff": ff,
                "drop_path": DropPath(dp)
            })
            block.gamma1 = nn.Parameter(1e-4 * torch.ones(dim))
            block.gamma2 = nn.Parameter(1e-4 * torch.ones(dim))
            self.layers.append(block)

    def forward(self, x):
        for block in self.layers:
            x = x + block["drop_path"](
                block.gamma1 * block["attn"](x)
            )
            x = x + block["drop_path"](
                block.gamma2 * block["ff"](x)
            )
        return x

#cell 4
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class CrossScaleAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 use_soma=False,
                 soma_topk=None,
                 soma_proj_query=True,
                 soma_proj_hidden=None,
                 soma_temperature=1.0):
        super().__init__()

        self.attn = Attention(
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
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.attn(
            self.norm(x)
        )

class MultiScaleTransformer(nn.Module):
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
                 pool='cls',
                 max_tokens=4096,
                 fusion_depth=2):
        super().__init__()

        self.pool = pool
        self.dim = dim
        self.fusion_depth = fusion_depth
        self.shared_transform = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            drop_path=0.1,
            use_soma=use_soma,
            soma_topk=soma_topk,
            soma_proj_query=soma_proj_query,
            soma_proj_hidden=soma_proj_hidden,
            soma_temperature=soma_temperature
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.cross_attn_blocks = nn.ModuleList([
            CrossScaleAttention(
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
            for _ in range(fusion_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, tokens_dict, H=None, W=None):
        th = self.shared_transform(tokens_dict['high'])
        tm = self.shared_transform(tokens_dict['mid'])
        tl = self.shared_transform(tokens_dict['low'])
        fused = torch.cat([th, tm, tl], dim=1)

        if self.fusion_depth == 0:
            fused = self.final_norm(fused)
            if self.pool == 'mean':
                return fused.mean(dim=1)
            return fused.mean(dim=1)
        B, N, D = fused.shape
        cls = repeat(
            self.cls_token,
            '1 1 d -> b 1 d',
            b=B
        )
        x = torch.cat([cls, fused], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]
        ]
        x = self.dropout(x)
        for block in self.cross_attn_blocks:
            x = block(x)
        x = self.final_norm(x)

        if self.pool == 'mean':
            return x.mean(dim=1)
        return x[:, 0]

class DNM(nn.Module):
    def __init__(self, in_channel, out_channel, num_branch=8, branch_hidden=64, dropout=0.1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_branch = num_branch
        self.input_norm = nn.LayerNorm(in_channel)
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channel,branch_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(branch_hidden,in_channel),
                nn.GELU()
            )
            for _ in range(num_branch)
        ])
        self.branch_attn = nn.Sequential(
            nn.Linear(in_channel,in_channel // 4),
            nn.GELU(),
            nn.Linear(in_channel // 4, 1)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(in_channel),
            nn.Linear(in_channel,in_channel),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_gate = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Sequential(
            nn.LayerNorm(in_channel),
            nn.Linear(in_channel,out_channel)
        )

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        branches = []
        for branch in self.branch_proj:
            branches.append(branch(x))
        branches = torch.stack(branches,dim=1)
        attn = self.branch_attn(branches)
        attn = torch.softmax(attn,dim=1)
        x = (branches * attn).sum(dim=1)
        x = self.fusion(x)
        x = x + self.res_gate * residual
        out = self.head(x)
        return out

#cell 5
import torch
from torch import nn

class SGMT(nn.Module):
    def __init__(self,
                 num_classes=100,
                 dim=256,
                 depth=12,
                 heads=12,
                 dim_head=16,
                 mlp_ratio=2,
                 base_patch=4,
                 use_conv_stem=True,
                 use_soma=True,
                 soma_topk=32,
                 fusion_depth=2,
                 head_type="dnm",
                 num_branch=8,
                 branch_hidden=128,
                 head_dropout=0.1):
        super().__init__()
        self.spt = MultiScaleSPT(
            in_dim=3,
            dim=dim,
            base_patch=base_patch,
            exist_class_t=False,
            use_conv_stem=use_conv_stem
        )
        self.transformer = MultiScaleTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=0.,
            use_soma=use_soma,
            soma_topk=soma_topk,
            soma_proj_query=True,
            soma_proj_hidden=None,
            soma_temperature=1.0,
            pool='cls',
            fusion_depth=fusion_depth
        )
        self.head_type = head_type
        if head_type == "dnm":
            self.head = DNM(
                in_channel=dim,
                out_channel=num_classes,
                num_branch=num_branch,
                branch_hidden=branch_hidden,
                dropout=head_dropout
            )
        elif head_type == "linear":
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim,num_classes)
            )
        else:
            raise ValueError(
                f"Unknown head_type: {head_type}"
            )

    def forward(self, x):
        tokens = self.spt(x)
        feat = self.transformer(tokens,H=x.shape[2],W=x.shape[3])
        out = self.head(feat)
        return out

#cell 6
import torch
import torch.optim as optim
import time, os, csv
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SGMT(
    num_classes=100,
    dim=256,
    depth=12,
    heads=12,
    dim_head=16,
    mlp_ratio=2,
    use_conv_stem=True,
    use_soma=True,
    soma_topk=32,
    fusion_depth=2,
    head_type="dnm"
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
EPOCHS = 100
warmup_epochs = 5

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)
    ],
    milestones=[warmup_epochs]
)
criterion = torch.nn.CrossEntropyLoss()
                
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.decay = decay

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                )

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
ema = EMA(model)

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam

def accuracy(output, target, topk=(1,5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1,1).expand_as(pred))
    return [(correct[:, :k].reshape(-1).float().sum(0) / target.size(0)) for k in topk]

SAVE_DIR = "/content/drive/MyDrive/SGMT_cifar100_no_DNM_checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)
ckpt_path = os.path.join(SAVE_DIR, "last.pth")
best_path = os.path.join(SAVE_DIR, "best.pth")
log_path = os.path.join(SAVE_DIR, "training_log.csv")
            
start_epoch = 0
best_acc = 0

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
            print("EMA loaded")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0)
        print(f"Resume from epoch {start_epoch}")
    else:
        model.load_state_dict(ckpt)
        print("Loaded legacy checkpoint")
else:
    print("Training from scratch")

if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_acc1","val_acc5","lr","time_s"])

for epoch in range(start_epoch, EPOCHS):
    t0 = time.time()
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        x, y_a, y_b, lam = mixup_data(x, y)
        optimizer.zero_grad()
        out = model(x)
        loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
        loss.backward()
        optimizer.step()
        ema.update()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    ema.apply_shadow()
    model.eval()
    acc1 = acc5 = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            a1, a5 = accuracy(out, y)
            acc1 += a1.item()
            acc5 += a5.item()

    acc1 /= len(test_loader)
    acc5 /= len(test_loader)
    ema.restore()
    lr = optimizer.param_groups[0]['lr']
    t = time.time() - t0
    print(f"Epoch {epoch+1:03d} | Loss {train_loss:.2f} | Acc@1 {acc1*100:.2f}% | Acc@5 {acc5*100:.2f}% | LR {lr:.6f}")
    with open(log_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, round(train_loss,2), round(acc1*100,2), round(acc5*100,2), round(lr,8), round(t,2)])

    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "epoch": epoch+1,
        "best_acc": best_acc
    }, ckpt_path)
    if acc1 > best_acc:
        best_acc = acc1
        torch.save(model.state_dict(), best_path)
    scheduler.step()
