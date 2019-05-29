# coding=utf8
"""
pytorch Dataset for regression test
use numpy generate it
"""
from torch.utils.data import Dataset
import torch


class DataRegression(Dataset):
    """
    class for generate data for pytorch regression
    """

    def __init__(self, bias=3.0, weights=torch.tensor([8, 2, 10], dtype=torch.float), sample_size=10000):
        data_x = torch.rand(weights.size()[0], sample_size, dtype=torch.float)
        data_y = torch.matmul(weights, data_x) + torch.randn(sample_size, dtype=torch.float) + bias
        print(data_y)
        self.data_x = data_x
        self.data_y = data_y
        self.sample_size = sample_size

    def __getitem__(self, item):
        return self.data_x[:, item], self.data_y[item]

    def __len__(self):
        return self.sample_size
