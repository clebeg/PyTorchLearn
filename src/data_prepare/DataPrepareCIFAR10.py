# coding=utf-8
"""
 @author: clebeg
 @contact: clebeg@163.com
 @site: https://my.oschina.net/u/1244232
 @file: DataPrepareCIFAR10.py
 @time: 2019-05-29 15:29
 @desc: parse cifar data can match image file from torchvision.datasetsImageFolder read format
 #from torchvision.datasets import ImageFolder
"""
import glob
import pickle
import os
import numpy as np
from scipy.misc import imsave
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def unpickle(data_file):
    with open(data_file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def save_data(base_path, unpickle_data, i):
    _img = np.reshape(unpickle_data[b'data'][i], (3, 32, 32))
    _img = _img.transpose(1, 2, 0)
    label_num = str(unpickle_data[b'labels'][i])
    # out_dir need mark label
    out_dir = os.path.join(base_path, label_num)
    mkdir(out_dir)
    # binary to str names decode
    img_name = os.path.join(out_dir, unpickle_data[b'filenames'][i].decode('utf8'))
    imsave(img_name, _img)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def prepare_train_data(base_path='/Users/clebeg/PycharmProjects/PyTorchLearn/datas/cifar-10-batches-py', train_per=0.8, valid_per=0.2):
    train_file_path = glob.glob1(base_path, 'data_batch_*')
    train_data_path = os.path.join(base_path, 'train_dataset')
    mkdir(train_data_path)
    valid_data_path = os.path.join(base_path, 'valid_dataset')
    mkdir(valid_data_path)
    for train_file in train_file_path:
        file_path = os.path.join(base_path, train_file)
        unpickle_data = unpickle(file_path)
        data_size = len(unpickle_data[b'labels'])
        shuffle = np.random.permutation(data_size)
        train_data_stop_ind = int(data_size*train_per)
        train_data_ind_list = shuffle[:train_data_stop_ind]
        test_data_ind_list = shuffle[train_data_stop_ind:]
        for i in train_data_ind_list:
            save_data(train_data_path, unpickle_data, i)
        for i in test_data_ind_list:
            save_data(valid_data_path, unpickle_data, i)
    return train_data_path, valid_data_path


if __name__ == '__main__':
    tp = '/Users/clebeg/PycharmProjects/PyTorchLearn/datas/cifar-10-batches-py/train_dataset'
    single_transform = transforms.Compose([transforms.ToTensor()])
    train_img_list = DataLoader(ImageFolder(tp, single_transform), batch_size=10)
    for img, label in train_img_list:
        print(img.size())
        print(label.size())
        break
