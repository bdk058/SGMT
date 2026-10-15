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
