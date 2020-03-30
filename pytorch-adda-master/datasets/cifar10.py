import torchvision
import torch
from torchvision import datasets, transforms
import params
'''
def get_cifar10(train):

    pre_process = transforms.Compose([transforms.ToTensor,
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))
                                      ])

    cifar10_dataset = datasets.CIFAR10(root=params.data_root,
                                       train=train,
                                       transform=pre_process,
                                       download=True)

    from datasets.modify_cifar_stl import modify_cifar, modify_cifar_t
    if train==train:
        modify_cifar(cifar10_dataset)
    else:
        modify_cifar_t(cifar_10_dataset)

    cifar10_data_loader = torch.utils.data.DataLoader(dataset=cifar10_dataset, batch_size = params.batch_size,shuffle=True)

    return cifar10_data_loader
'''
def get_cifar10(train):
    tr_dataset = torchvision.datasets.CIFAR10(root=params.data_root, train=True, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))]))
    te_dataset = torchvision.datasets.CIFAR10(root=params.data_root, train=False, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))]))
    from datasets.modify_cifar_stl import modify_cifar, modify_cifar_t
    modify_cifar(tr_dataset)
    modify_cifar_t(te_dataset)
    
    if train==True:
        cifar10_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size = params.batch_size, shuffle=True)
    else:
        cifar10_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size = params.batch_size, shuffle=True)

    return cifar10_data_loader
