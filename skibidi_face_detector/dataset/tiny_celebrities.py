import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

ds = load_dataset("brendenc/celeb-identities").with_format('torch')['train']
num_classes = len(set(ds['label']))

transformer = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
])


def transform(example):
    example['image'] = transformer(example['image'])
    return example


ds = ds.map(transform, batched=False)

split = ds.train_test_split(test_size=0.2)
train_ds, test_ds = split['train'], split['test']

train_loader = DataLoader(train_ds, shuffle=True)
test_loader = DataLoader(test_ds, shuffle=False)
