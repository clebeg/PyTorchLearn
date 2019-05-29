# coding=utf-8
"""
 @author: clebeg
 @contact: clebeg@163.com
 @site: https://my.oschina.net/u/1244232
 @file: ResNetFineTune.py
 @time: 2019-05-29 20:47
 @desc: use already trained resnet model finetune on cifar data
"""
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch


def reset_resnet_model(label_num=10):
    model_fn = resnet18(pretrained=True)
    fc_in_features = model_fn.fc.in_features
    model_fn.fc = Linear(fc_in_features, label_num)
    return model_fn


def data_generate():
    train_path = '/Users/clebeg/PycharmProjects/PyTorchLearn/datas/cifar-10-batches-py/train_dataset'
    valid_path = '/Users/clebeg/PycharmProjects/PyTorchLearn/datas/cifar-10-batches-py/valid_dataset'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，再随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像50%的概率翻转，50%的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = DataLoader(ImageFolder(train_path, transform_train), batch_size=256, shuffle=True, num_workers=1)
    valid_data = DataLoader(ImageFolder(valid_path, transform_valid), batch_size=128, num_workers=1)

    dataset_loader = {'train': train_data, 'valid': valid_data}
    dataset_size = {'train': len(train_data.dataset.imgs),
                    'valid': len(valid_data.dataset.imgs)}
    return dataset_loader, dataset_size


def train_and_valid(resnet_model, loss_fn, optim_fn,
                    learn_rate_scheduler, dataset_loader,
                    dataset_size, task_type, best_model):
    # each epoch need train and valid
    if task_type == 'train':
        learn_rate_scheduler.step()
        resnet_model.train(True)
    else:
        resnet_model.train(False)
    epoch_sum_loss = 0.0
    epoch_sum_corrects = 0
    batch_num = 0
    for input_data, input_label in dataset_loader[task_type]:
        optim_fn.zero_grad()
        model_pred = resnet_model(input_data)
        loss = loss_fn(model_pred, input_label)
        _, pred_label = torch.max(model_pred, 1)
        if task_type == 'train':
            loss.backward()
            optim_fn.step()
        epoch_sum_loss += loss.data.item()
        epoch_sum_corrects += torch.sum(pred_label == input_label).data.item()
        progress = dataset_loader[task_type].batch_size*batch_num/dataset_size[task_type]
        print('\r{} Progress {:2.1%}'.format(task_type, progress), end='', flush=True)
        batch_num += 1
    print('\r{} Progress 100%'.format(task_type), flush=True)
    epoch_loss = epoch_sum_loss/dataset_size[task_type]
    epoch_corrects = epoch_sum_corrects/dataset_size[task_type]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(task_type, epoch_loss, epoch_corrects))

    if task_type == 'valid' and epoch_corrects > best_model['best_acc']:
        best_model['best_acc'] = epoch_corrects
        best_model['best_loss'] = epoch_loss
        best_model['best_model'] = resnet_model.state_dict()


def finetune_resnet(resnet_model, dataset_loader, dataset_size, num_epochs=25):
    learn_rate = 0.01
    # momentum = 0.9
    # define loss function, class model need cross entropy
    loss_fn = torch.nn.CrossEntropyLoss()
    # define optim function, here use sgd
    optim_fn = torch.optim.Adam(resnet_model.parameters(), lr=learn_rate)
    # define learn rate will decrease by each step size
    learn_rate_scheduler = torch.optim.lr_scheduler.StepLR(optim_fn, step_size=7, gamma=0.1)
    best_model = {
        'best_loss': 0.0,
        'best_acc': 0.0,
        'best_model': resnet_model.state_dict()
    }
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs))
        print('--'*20)
        train_and_valid(resnet_model, loss_fn, optim_fn,
                        learn_rate_scheduler, dataset_loader,
                        dataset_size, 'train', best_model)
        train_and_valid(resnet_model, loss_fn, optim_fn,
                        learn_rate_scheduler, dataset_loader,
                        dataset_size, 'valid', best_model)
        print()
    return best_model


def main():
    rs_model = reset_resnet_model(10)
    ds_loader, ds_size = data_generate()
    finetune_resnet(rs_model, ds_loader, ds_size, 25)


if __name__ == '__main__':
    main()

