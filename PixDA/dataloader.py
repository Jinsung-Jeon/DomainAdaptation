import torch
from torchvision import datasets, transforms
from DsetThreeChannels import ThreeChannels
import torchvision

def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root="data",
                                   train=train,
                                   transform=pre_process,
                                   download=True)
    
    mnist_dataset = ThreeChannels(mnist_dataset)
    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=96,
        shuffle=True)

    return mnist_data_loader

def get_svhn(split='train'):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))])

    svhn_dataset = datasets.SVHN(root="data",split='train', transform=pre_process, download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=96,
        shuffle=True)

    return svhn_data_loader

def get_cifar10(train):
    tr_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))]))
    te_dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))]))
    from modify_cifar_stl import modify_cifar, modify_cifar_t
    modify_cifar(tr_dataset)
    modify_cifar_t(te_dataset)

    if train==True:
        cifar10_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size = 128, shuffle=True)
    else:
        cifar10_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size = 128, shuffle=True)

    return cifar10_data_loader

import torch
from torchvision import datasets, transforms

def get_stl10(split='train'):
    pre_process = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))])
    stl10_dataset = datasets.STL10(root="data", split='train', transform=pre_process, download=True)

    from modify_cifar_stl import modify_stl
    modify_stl(stl10_dataset)

    stl_data_loader = torch.utils.data.DataLoader(
        dataset=stl10_dataset,
        batch_size=128,
        shuffle=True)

    return stl_data_loader
