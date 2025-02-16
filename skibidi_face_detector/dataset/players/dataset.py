from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import os

transformer = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')

ds = ImageFolder(root, transform=transformer)
class_to_idx = ds.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

loader = DataLoader(ds, shuffle=True)
