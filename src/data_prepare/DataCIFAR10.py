# coding=utf8
"""
prepare data cifar10 for training
1. auto download data from network
2. unzip data
"""
from utils import file_util
import pickle
import numpy as np
import os
from scipy.misc import imsave
import random


cifar_10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' # need dowload file
data_dir = '/Users/clebeg/PycharmProjects/PyTorchLearn/datas/' # local data root path
local_file_dir = os.path.join(data_dir, 'cifar-10-python.tar.gz') # local file name
uncompress_data_dir = os.path.join(data_dir, 'cifar-10-batches-py') # uncompress data dir
test_data_file = os.path.join(uncompress_data_dir, 'test_batch')


train_data_dir = os.path.join(data_dir, 'raw_train')
test_data_dir = os.path.join(data_dir, 'raw_test')

train_file_names = os.path.join(data_dir, 'train.txt')
valid_file_names = os.path.join(data_dir, 'valid.txt')
test_file_names = os.path.join(data_dir, 'test.txt')

train_percent = 0.8
valid_percent = 0.1
test_percent = 0.1


def download_uncompress_cifar_10(local_file):
    file_util.wget_network_file(cifar_10_url, local_file)
    file_util.uncompress(local_file)


def unpickle(data_file):
    with open(data_file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def mkdir(data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)


def prepare_data(data_file, out_path):
    unpickle_data = unpickle(data_file)
    print(data_file + ' is loading ...')
    for i in range(0, len(unpickle_data[b'labels'])):
        img = np.reshape(unpickle_data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        label_num = str(unpickle_data[b'labels'][i])
        # out_dir need mark label
        out_dir = os.path.join(out_path, label_num)
        mkdir(out_dir)
        # binary to str names decode
        img_name = os.path.join(out_dir, label_num + '_' + unpickle_data[b'filenames'][i].decode('utf8'))
        imsave(img_name, img)
    print(data_file + ' loaded.')


def prepare_all_data():
    # prepare train data
    for j in range(1, 6):
        train_data_file = os.path.join(uncompress_data_dir, 'data_batch_' + str(j))
        prepare_data(train_data_file, train_data_dir)
    prepare_data(test_data_file, test_data_dir)


def split_data():
    # split the data: train valid test only use
    unpickle_data = unpickle(test_data_file)
    names_and_label = []
    for i in range(0, len(unpickle_data[b'labels'])):
        label_num = str(unpickle_data[b'labels'][i])
        names = label_num + '_' + os.path.join(test_data_dir, label_num + '_' + unpickle_data[b'filenames'][i].decode('utf8'))
        names_and_label.append(names + ' ' + label_num)
    random.seed(666)
    random.shuffle(names_and_label)
    train_ind = int(len(names_and_label) * train_percent)
    valid_ind = int(len(names_and_label) * (train_percent + valid_percent))
    with open(train_file_names, 'w') as f:
        f.write('\n'.join(names_and_label[:train_ind]))
    with open(valid_file_names, 'w') as f:
        f.write('\n'.join(names_and_label[train_ind:valid_ind]))
    with open(test_file_names, 'w') as f:
        f.write('\n'.join(names_and_label[valid_ind:]))


if __name__ == '__main__':
    # prepare_all_data()
    split_data()
