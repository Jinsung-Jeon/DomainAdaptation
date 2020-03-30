import torch
from torchvision import datasets, transforms
import params

def get_stl10(train, split='train'):
    pre_process = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))])
    stl10_dataset = datasets.STL10(root=params.data_root, split='train', transform=pre_process, download=True)

    from datasets.modify_cifar_stl import modify_stl
    modify_stl(stl10_dataset)

    stl_data_loader = torch.utils.data.DataLoader(
        dataset=stl10_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return stl_data_loader
