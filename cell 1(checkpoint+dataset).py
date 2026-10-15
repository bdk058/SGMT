from google.colab import drive
drive.mount('/content/drive')

import os

SAVE_DIR = "/content/drive/MyDrive/SGMT_cifar100_no_DNM_checkpoint"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("Checkpoint dir:", SAVE_DIR)

!git clone https://github.com/cyizhuo/CIFAR-100-dataset.git

DATA_DIR = "./CIFAR-100-dataset"
