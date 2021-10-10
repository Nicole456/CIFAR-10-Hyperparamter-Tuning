from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch


def cifar10_dataset(dataset_base_path, train_flag=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    if train_flag:
        transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]
        )
    # import pdb;pdb.set_trace()
    dataset = datasets.CIFAR10(root=dataset_base_path, train=train_flag,
                               download=False, transform=transform)
    return dataset


def get_sl_sampler(labels, valid_num_per_class, num_classes):
    """
    :param labels: torch.array(int tensor)
    :param valid_num_per_class: the number of validation for each class
    :param num_classes: the total number of classes
    :return: sampler_l,sampler_u
    """
    sampler_valid = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc=loc.view(-1)
        loc=loc[torch.randperm(loc.size(0))]
        sampler_valid.extend(loc[:valid_num_per_class].tolist())
        sampler_train.extend(loc[valid_num_per_class:].tolist())
    sampler_valid=SubsetRandomSampler(sampler_valid)
    sampler_train=SubsetRandomSampler(sampler_train)
    return sampler_train,sampler_valid