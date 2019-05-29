# encoding=utf8
from torch.utils.data import Dataset
from PIL import Image


class DatasetCIFAR10(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        with open(txt_path, 'r') as f:
            imgs = [line.strip().split() for line in f.readlines(txt_path)]
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = Image.open(fn).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.imgs)
