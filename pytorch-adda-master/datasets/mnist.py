"""Dataset setting and data loader for MNIST."""
import torch
from torchvision import datasets, transforms
from datasets.DsetThreeChannels import ThreeChannels
import params


def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_dataset = ThreeChannels(mnist_dataset)

    mnist_data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return mnist_data_loader

def get_mnist_set(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    mnist_dataset = ThreeChannels(mnist_dataset)

    return mnist_dataset

