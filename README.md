# PyTorchLearn
目标通过本项目学习 PyTorch，实战深度学习，同时也为想通过 PyTorch 学习深度学习的朋友积累经验，欢迎一起交流

## 一、 PyTorch 实战深度学习
位于 jupyter_script/dl_in_action 目录，将来还会增加其他学习 jupyter script

下面是具体的学习目录
    
一、深度学习基础
+ 1.1 [PyTorch基本数值操作](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/PyTorch%E5%9F%BA%E6%9C%AC%E6%95%B0%E5%80%BC%E6%93%8D%E4%BD%9C.ipynb)
+ 1.2 [PyTorch自动求梯度](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/PyTorch%E8%87%AA%E5%8A%A8%E6%B1%82%E6%A2%AF%E5%BA%A6.ipynb)
+ 1.3 [线性回归](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.ipynb)
+ 1.4 [逻辑回归](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/逻辑回归.ipynb)
+ 1.5 [softmax回归](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/softmax回归.ipynb)
+ 1.6 [多层感知机](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/多层感知机.ipynb)
+ 1.7 [过拟合与欠拟合](https://github.com/clebeg/PyTorchLearn/blob/master/jupyter_script/dl_in_action/过拟合与欠拟合.ipynb)

## 二、拿来即用模型
位于 src 目录中

### 2.1 项目全局配置
```src.config.constant```
+ DATA_ROOT_PATH: 训练数据集下载目录

数据将下载到对应的本地目录中，这部分数据不会上传到 git

### 2.2 使用的数据集
将下面数据集下载到 DATA_ROOT_PATH 目录中
+ cifar-10-python.tar.gz: 经典图片分类数据集，总共10个类别，下载地址：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz，官方网站：https://www.cs.toronto.edu/~kriz/cifar.html
+ THUCNews_model.zip: 清华大学自然语言处理实验室提供的中文语料，官网地址：http://thuctc.thunlp.org/#中文文本分类数据集THUCNews，需要提供用户名、公司名和邮箱即可下载，总共有14个类别，并提供了实验室的测试效果
