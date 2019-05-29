# coding=utf8
"""
build linear regression model
"""
from torch.nn import Module
from torch.nn import Linear
from torch.utils.data import DataLoader
from src.data_prepare.DataRegression import DataRegression
import torch

total_epoch = 10
batch_size = 10
learn_rate = 0.05
batch_print = 2


class LinearRegression(Module):
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()
        self.fc = Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class RawLinearRegression:
    def __init__(self, in_features, out_features):
        self.weights = torch.randn(in_features, dtype=torch.float)  # 3*1
        self.bias = torch.randn(out_features, dtype=torch.float)  # 1*1

    def forward(self, x):
        # 记住一定要加偏置项
        pred_y = torch.matmul(x, self.weights) + self.bias  # 100*3 3*1
        return pred_y

    @staticmethod
    def loss(pred_y, true_y):
        loss = torch.mean(torch.pow(pred_y - true_y, 2))
        return torch.sqrt(loss)

    def sgd(self, pred_y, true_y, true_x, lr=0.01):
        sample_size = true_y.size()[0]
        weights_grad = torch.matmul(true_y - pred_y, true_x)
        self.weights.add_(lr*weights_grad/sample_size)
        bias_grad = torch.sum(true_y - pred_y)
        self.bias.add_(lr*bias_grad/sample_size)


def train_raw():
    # prepare train data 准备数据集
    regression_helper = RawLinearRegression(3, 1)
    data_regression = DataRegression()
    train_data_loader = DataLoader(data_regression, batch_size)
    train_loss = []
    for i in range(1, total_epoch + 1):
        for data_x, data_y in train_data_loader:
            pred_y = regression_helper.forward(data_x)
            loss = RawLinearRegression.loss(pred_y, data_y)
            regression_helper.sgd(pred_y, data_y, data_x, learn_rate)
            train_loss.append(loss)
        if i % batch_print == 0:
            print("Loss = ", train_loss[-1])

    print(regression_helper.weights, regression_helper.bias)


def train_auto_grad():
    # prepare train data 准备数据集
    data_regression = DataRegression()
    train_data_loader = DataLoader(data_regression, batch_size)

    # prepare loss function 定义损失需要有括号
    loss_fn = torch.nn.MSELoss()

    # prepare model
    # linear_regression = LinearRegression(4, 1)
    linear_regression = LinearRegression(3, 1)

    # optimizer 优化器需要接受模型的参数
    optimizer = torch.optim.SGD(linear_regression.parameters(), lr=learn_rate)

    train_loss = []
    loss = None
    for i in range(1, total_epoch+1):
        for ind, batch_data in enumerate(train_data_loader):
            data_x = batch_data[0]
            data_y = batch_data[1].view(1, -1)
            # 注意顺序 优化器梯度归零 自动会将模型梯度归零
            optimizer.zero_grad()

            # 损失反向传播 用框架定义的损失函数
            predict_y = linear_regression(data_x).view(1, -1)
            loss = loss_fn(predict_y, data_y)
            loss.backward()

            # 优化器走一步 更新模型的参数
            optimizer.step()
        train_loss.append(loss)
        if i % batch_print == 0:
            print("Loss = ", loss)
    for f in linear_regression.parameters():
        print(f)

    # 在验证集的效果
    

if __name__ == '__main__':
    train_raw()
