import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from calibrate import *
# from models import resnet110

from pdb import set_trace as trace

# from timm.models import
# from resnet import *
# import resnet1
from map_labels import *



train_excluded_labels = list(range(6, 10))  # [6,7,8,9]
test_excluded_labels = list(range(6))  # [0,1,2,3,4,5]


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet110')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--num_cuda', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4)
    parsed, _ = parser.parse_known_args()
    return parsed


# preprocess image dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(96),
    transforms.ToTensor(),
    #     transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96),
    #    transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])


def get_train_dataset(dataset, transform, root, target):
    # trace()
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=True, transform=transform, download=False)
    else:
        return dataset(root=root, split='train', transform=transform, download=False)


def get_test_dataset(dataset, transform, root, target):
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=False, transform=transform, download=False)
    else:
        return dataset(root=root, split='test', transform=transform, download=False)


# def get_train_loader(target='cifar100', root='./data'):
def get_train_loader(target='cifar100', root='E:/Datasets/CIFAR10'):
    """ return dataloader for image dataset

    Args:
        target: name of the dataset to be loaded
        root: path to the dataset directory
        train: whether to use train data

    Return:
        dataloader: dataset loader object
    """
    # set target dataset
    if target == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
    elif target == 'cifar100':
        dataset = torchvision.datasets.CIFAR100
    elif target == 'svhn':
        dataset = torchvision.datasets.SVHN
    elif target == 'mnist':
        dataset = torchvision.datasets.MNIST

    transform = transform_train
    shuffle = True
    batch_size = 128

    # dataset = dataset(root=root, train=True, transform=transform, download=False)
    # train_idxs = get_idxs_without_excluded(dataset.targets, train_excluded_labels)

    dataset = get_train_dataset(dataset, transform, root, target)
    labels = get_labels(dataset)
    train_idxs = get_idxs_without_excluded(get_labels(dataset), train_excluded_labels)
    train_modified = torch.utils.data.Subset(dataset, train_idxs)

    # train, val = torch.utils.data.random_split(train_modified, [27000, 3000])
    train, val = torch.utils.data.random_split(train_modified, [int(0.9 * len(train_idxs)),
                                                                len(train_idxs) - int(0.9 * len(train_idxs))])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2)

    return train_loader, val_loader


# def get_test_loader(target='cifar100', root='./data'):
def get_test_loader(target='cifar100', root='E:/Datasets/CIFAR10'):
    """ return dataloader for image dataset

    Args:
        target: name of the dataset to be loaded
        root: path to the dataset directory
        train: whether to use train data

    Return:
        dataloader: dataset loader object
    """
    # set target dataset
    if target == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
    elif target == 'cifar100':
        dataset = torchvision.datasets.CIFAR100
    elif target == 'svhn':
        dataset = torchvision.datasets.SVHN
    elif target == 'mnist':
        dataset = torchvision.datasets.MNIST

    transform = transform_test
    shuffle = False
    batch_size = 128

    dataset = get_test_dataset(dataset, transform, root, target)

    #    test_idxs = get_idxs_without_excluded(dataset.targets, train_excluded_labels)
    #    test_modified = torch.utils.data.Subset(dataset, test_idxs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=2)

    return loader


def get_model(name, num_classes, num_channels):
    """ return model architecture for image recognition

    Args:
        name: which model architecture to use
        num_classes: number of classes for classification
        pretrained: whether or not to get imagenet pretrained model

    Return:
        model: neural network object to be used
    """
    if name == 'resnet18':
        # model = models.resnet18
        model = resnet18 if num_channels == 3 else resnet1.resnet18
    elif name == 'resnet34':
        # model = models.resnet34
        model = resnet34 if num_channels == 3 else resnet1.resnet34
    elif name == 'resnet50':
        # model = models.resnet50
        model = resnet50 if num_channels == 3 else resnet1.resnet50
    elif name == 'resnet101':
        # model = models.resnet101
        model = resnet101 if num_channels == 3 else resnet1.resnet101
    elif name == 'resnet110':
        model = resnet110 if num_channels == 3 else resnet1.resnet110
    elif name == 'resnet152':
        # model = models.resnet152
        model = resnet152 if num_channels == 3 else resnet1.resnet152

    return model(num_classes=num_classes)


def get_optim(name, model):
    """ return optimizer instance for training

    Args:
        name: which optimizer to use
        model: neural network to be optimized

    Return:
        optimizer: optimizer object
        scheduler: learning rate scheduler for optimizer
    """
    if name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[60, 120, 160, 200], gamma=0.2)
        return optimizer, scheduler
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4,
                              momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[60, 120, 160, 200], gamma=0.2)
        return optimizer, scheduler


def get_calibrator(logits, labels, mode):
    """ return calibrator object that generates confidences

    Args:
        logits: logits that will be used for training calibrator
        labels: lables that will be used for training calibrator
        mode: type of calibrator to be employed

    Return:
        function: a calibrator object
    """
    if mode == 'histogram_binning':
        return multiclass_binning(logits, labels)
    elif mode == 'matrix_scaling':
        return matrix_scaling(logits, labels)
    elif mode == 'vector_scaling':
        return vector_scaling(logits, labels)
    elif mode == 'temperature_scaling':
        return temperature_scaling(logits, labels)
