# coding=utf8
"""
some file utils function collect here
"""
from urllib.request import urlopen
from urllib.request import Request
import gzip
import tarfile
import zipfile
import os
import subprocess


def download_network_file(url, local_file):
    """
    download file from network
    :param url: network file need download
    :param local_file: local file for store download data
    :return: boolean(True:download success,False: download error)
    """
    if os.path.exists(local_file):
        return True
    line = 1
    res = urlopen(url)  # type: Request
    chunk = 1024*16
    with open(local_file, 'wb') as f:
        while True:
            if line % 100 == 0:
                print('.')
            else:
                print('.', end='')
            buffer = res.read(chunk)
            if not buffer:
                break
            line += 1
            f.write(buffer)
    print('Download Complete!')
    if not os.path.exists(local_file):
        print('{} Download Error, {} Not Exist!'.format(url, local_file))
        return False
    return True


def wget_network_file(url, local_file):
    if os.path.exists(local_file):
        return True
    download_process = subprocess.Popen(['wget', '-c', '-O', local_file, "{}".format(url)])
    download_process.wait()
    if not os.path.exists(local_file):
        print('{} Download Error, {} Not Exist!'.format(url, local_file))
        return False
    return True


def uncompress(src_file, target_dir=None, file_fmt='tgz'):
    if not target_dir:
        target_dir = os.path.dirname(os.path.abspath(src_file))
    if file_fmt in ('tgz', 'tar'):
        tar = tarfile.open(src_file)
        names = tar.getnames()
        for name in names:
            tar.extract(name, target_dir)
        tar.close()
    elif file_fmt == 'zip':
        zip_file = zipfile.ZipFile(src_file)
        for names in zip_file.namelist():
            zip_file.extract(names, target_dir)
        zip_file.close()
    else:
        return False
    return True


if __name__ == '__main__':
    test_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    test_local_file = '/Users/clebeg/PycharmProjects/PyTorchLearn/datas/cifar-10-python.tar.gz'
    #print(wget_network_file(test_url, test_local_file))
    print(uncompress(test_local_file))
