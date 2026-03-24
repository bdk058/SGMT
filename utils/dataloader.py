import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import random

from utils.data_aug import CIFARPolicy, RandomErasing


# --------------------------
# 🔹 Custom Dynamic Collate
# --------------------------
def dynamic_collate(batch):
    """
    Custom collate_fn cho phép batch chứa ảnh nhiều kích thước khác nhau.
    Tự động pad ảnh nhỏ hơn lên cùng kích thước lớn nhất trong batch.
    """
    imgs, labels = zip(*batch)

    # Lấy chiều cao / rộng lớn nhất trong batch
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)

    padded_imgs = []
    for img in imgs:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
        img_padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
        padded_imgs.append(img_padded)

    imgs_tensor = torch.stack(padded_imgs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return imgs_tensor, labels_tensor


# --------------------------
# 🔹 Repeated Augmentation Sampler (giữ nguyên logic gốc)
# --------------------------
class RASampler(Sampler):
    def __init__(self, dataset_len, batch_size, repetitions=1, len_factor=3.0, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def shuffler(self):
        if self.shuffle:
            new_perm = lambda: iter(np.random.permutation(self.dataset_len))
        else:
            new_perm = lambda: iter(np.arange(self.dataset_len))
        shuffle = new_perm()
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            for _ in range(self.repetitions):
                yield index

    def __iter__(self):
        shuffle = iter(self.shuffler())
        batch = []
        for _ in range(self.len_images):
            index = next(shuffle)
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.len_images // self.batch_size
        else:
            return (self.len_images + self.batch_size - 1) // self.batch_size


# --------------------------
# 🔹 Dynamic CIFAR Loader (DVT-V2)
# --------------------------
class CifarLoader:
    def __init__(self, config):
        self.data_root = os.path.join(config['data_root'], config['cifar_type'])
        self.cifar_type = config['cifar_type']
        self.valid_scale = config['valid_scale']
        self.batch_size = config['batch_size']
        self.norm = config['norm']
        self.mp = config['multi_process']
        self.num_classes = config['num_classes']

        # Augmentation (không Resize nữa!)
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),  # crop vẫn hoạt động nếu ảnh đủ lớn
        ]

        if 'augmentation' in config:
            aug_config = config['augmentation']
            if 'aug_policy' in aug_config and aug_config['aug_policy'] == 'CIFAR':
                aug.append(CIFARPolicy())
            aug.append(transforms.ToTensor())
            aug.append(transforms.Normalize(self.norm[0], self.norm[1]))

            if 'random_erasing' in aug_config:
                re_config = aug_config['random_erasing']
                re = RandomErasing(
                    probability=re_config['prob'],
                    sh=re_config['sh'],
                    r1=re_config['r1'],
                    mean=self.norm[0],
                )
                aug.append(re)
        else:
            aug = [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1])
            ]

        train_transform = transforms.Compose(aug)
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1])
        ])

        # Dataset
        if self.cifar_type == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(
                root=self.data_root, train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(
                root=self.data_root, train=False, download=True, transform=val_test_transform)
        else:
            trainset = torchvision.datasets.CIFAR100(
                root=self.data_root, train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR100(
                root=self.data_root, train=False, download=True, transform=val_test_transform)

        # Split train/valid
        if self.valid_scale > 0:
            train_size = int(len(trainset) * (1 - self.valid_scale))
            val_size = len(trainset) - train_size
            train_data, valid_data = random_split(trainset, [train_size, val_size])
        else:
            train_data, valid_data = trainset, testset

        # Dataloader (sử dụng dynamic_collate)
        self.trainloader = DataLoader(
            train_data,
            batch_sampler=RASampler(len(train_data), self.batch_size, 1, 3.0, shuffle=True, drop_last=True),
            num_workers=self.mp,
            pin_memory=True,
            collate_fn=dynamic_collate
        )

        self.validloader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mp,
            pin_memory=True,
            collate_fn=dynamic_collate
        )

        self.testloader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.mp,
            pin_memory=True,
            collate_fn=dynamic_collate
        )

        print("[DVT-V2] Dynamic DataLoader initialized — multi-size batch enabled.")
