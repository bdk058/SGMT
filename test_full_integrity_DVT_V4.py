# test_full_integrity_DVT_V4.py
import json, torch
from network.network import DVT_V4

cfg_all = json.load(open("/content/DVT/configs/DVT_cifar10.json"))
cfg = cfg_all["network_params"]
# ensure name
cfg['name'] = 'DVT_V4'
model = DVT_V4(cfg, device="cpu").eval()

for H,W in [(32,32),(48,48),(64,64)]:
    x = torch.randn(2,3,H,W)
    out = model(x)
    print(f"{H}x{W} ->", out.shape)

# backward check
x = torch.randn(2,3,32,32, requires_grad=True)
out = model(x)
(out.sum()).backward()
print("backward ok")
