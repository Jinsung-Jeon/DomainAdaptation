import torch
from torchvision import datasets, transforms
from DsetThreeChannels import ThreeChannels

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
