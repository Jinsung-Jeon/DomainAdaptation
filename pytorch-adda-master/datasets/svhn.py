import torch
from torchvision import datasets, transforms
import params


def get_svhn(train, split='train'):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))])

    svhn_dataset = datasets.SVHN(root=params.data_root,split='train', transform=pre_process, download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return svhn_data_loader

def get_svhn_set(train, split='train'):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.437, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))])

    svhn_dataset = datasets.SVHN(root=params.data_root,split='train', transform=pre_process, download=True)

    return svhn_dataset
