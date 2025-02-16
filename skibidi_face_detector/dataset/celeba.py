import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from datasets import load_dataset

ds = load_dataset("flwrlabs/celeba").with_format("torch")

train_ds = ds['train']
cols = train_ds.column_names
cols.remove('image')
cols.remove('celeb_id')
train_ds = train_ds.remove_columns(cols).rename_column('celeb_id', 'label')

train_num_classes = len(set(train_ds['label']))

test_ds = ds['test']
cols = test_ds.column_names
cols.remove('image')
cols.remove('celeb_id')
test_ds = test_ds.remove_columns(cols).rename_column('celeb_id', 'label')

test_num_classes = len(set(test_ds['label']))

val_ds = ds['valid']
cols = val_ds.column_names
cols.remove('image')
cols.remove('celeb_id')
val_ds = val_ds.remove_columns(cols).rename_column('celeb_id', 'label')

val_num_classes = len(set(val_ds['label']))

BATCH_SIZE = 32

train_loader = DataLoader(train_ds, shuffle=True, num_workers=11, batch_size=BATCH_SIZE, persistent_workers=True)
test_loader = DataLoader(test_ds, shuffle=False, num_workers=11, batch_size=BATCH_SIZE, persistent_workers=True)
val_loader = DataLoader(val_ds, shuffle=False, num_workers=11, batch_size=BATCH_SIZE, persistent_workers=True)

augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
])

transformer = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
])

split = val_ds.train_test_split(test_size=0.2)
val_train_ds, val_test_ds = split['train'], split['test']

val_train_loader = DataLoader(val_train_ds, shuffle=True, num_workers=11, batch_size=BATCH_SIZE, persistent_workers=True)
val_test_loader = DataLoader(val_test_ds, shuffle=False, num_workers=11, batch_size=BATCH_SIZE, persistent_workers=True)

