'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST, VisionDataset, SVHN
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split


class CustomMNIST(VisionDataset):
    training_file = 'training.pt'
    testing_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, classes=[0, 1]):
        super(CustomMNIST, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
        assert len(classes) == 2, "Code was only implemented for 2 class MNIST, will need modifications"
        self.train = train
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.testing_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        select_indexes = [i for i, x in enumerate(self.targets.tolist()) if x in classes]
        #         targets = [-1 for i in select_indexes if self.targets[i]==classes[0] else 1]
        self.data, self.targets = self.data[select_indexes], self.targets[select_indexes]
        self.targets[self.targets == classes[0]] = 1.0
        self.targets[self.targets == classes[1]] = -1.0
        self.targets = self.targets.float()

    #         self.targets = sel

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

## Added:
import os

def progress_bar(current, total, message='', bar_length=40):
    percent = float(current) / total
    arrow_length = int(bar_length * percent)
    spaces = ' ' * (bar_length - arrow_length)

    if message:
        progress_bar = '[' + '=' * arrow_length + '>' + '.' * (bar_length - arrow_length) + ']'
        print('\r{} {}% {}'.format(progress_bar, int(percent * 100), message), end='')
    else:
        progress_bar = '[' + '=' * arrow_length + '>' + '.' * (bar_length - arrow_length) + '] {:.2f}%'.format(
            percent * 100)
        print('\r{}'.format(progress_bar), end='')

    if current == total:
        print()

def get_terminal_width():
    try:
        _, term_width = os.get_terminal_size()
    except OSError:
        term_width = 80  # default width if the terminal size cannot be obtained
    return term_width

##

def get_loaders_mnist(classes, batch_size):
    # MNIST dataset
    train_dataset = CustomMNIST(root='../datasets/',
                                train=True,
                                transform=transforms.ToTensor(),
                                classes=classes)

    test_dataset = CustomMNIST(root='../datasets/',
                               train=False,
                               transform=transforms.ToTensor(),
                               classes=classes)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    return train_loader, test_loader

def get_loaders_mnist_full(batch_size, images_use):
    # MNIST dataset
    train_dataset = MNIST(root='../datasets/',
                                train=True,
                                transform=transforms.ToTensor(),
                                )

    test_dataset = MNIST(root='../datasets/',
                               train=False,
                               transform=transforms.ToTensor(),
                               )
    # ipdb.set_trace()
    train_dataset, val_dataset = random_split(train_dataset, (images_use, 60000-images_use))
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def get_loaders_fashion_mnist_full(batch_size, images_use):
    # Fashion MNIST dataset
    train_dataset = FashionMNIST(root='../datasets/',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor(),
                                )

    test_dataset = FashionMNIST(root='../datasets/',
                               train=False,
                               download=True,
                               transform=transforms.ToTensor(),
                               )
    # ipdb.set_trace()
    train_dataset, val_dataset = random_split(train_dataset, (images_use, 60000-images_use))
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader

def get_loaders_svhn(batch_size, normalize=True):
    if normalize:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    trainset = SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def get_loaders_cifar(batch_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
    ])
    trainset = CIFAR10(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = CIFAR10(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


