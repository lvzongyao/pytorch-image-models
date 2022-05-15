import argparse

import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
# from resnet import *
import torchvision.transforms as transforms

from calibrate import *
# from models import resnet110


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet110')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--num_epoch', type=int, default=30)
    parser.add_argument('--num_cuda', type=int, default=0)
    parsed, _ = parser.parse_known_args()
    return parsed


args = argparser()

# preprocess image dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(96),
    transforms.ToTensor(),
    # transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96),
    # transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])


def get_train_dataset(dataset, transform, root, target):
    # trace()
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=True, transform=transform, download=False)
    elif target == 'svhn':
        return dataset(root=root, split='train', transform=transform, download=False)
    else:
        print("dataset doesn't exist")


def get_test_dataset(dataset, transform, root, target):
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=False, transform=transform, download=False)
    elif target == 'svhn':
        return dataset(root=root, split='train', transform=transform, download=False)
    else:
        print("dataset doesn't exist")


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
    # batch_size = 256

    dataset = get_train_dataset(dataset, transform, root, target)
    # if args.dataset == 'cifar10' or args.dataset == 'mnist':
    #     train, val = torch.utils.data.random_split(dataset, [45000, 5000])
    # if args.dataset == 'svhn':
    #     train, val = torch.utils.data.random_split(dataset, [65931, 7326])
    train, val = torch.utils.data.random_split(dataset, [int(0.9 * len(dataset)),
                                                                len(dataset) - int(0.9 * len(dataset))])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=2, collate_fn=fast_collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2, collate_fn=fast_collate)

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
    # batch_size = 100
    batch_size = 128
    # batch_size = 256
    # batch_size = 200

    dataset = get_test_dataset(dataset, transform, root, target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=2, collate_fn=fast_collate)
    return loader


# def get_model(name, num_classes):
#     """ return model architecture for image recognition
#
#     Args:
#         name: which model architecture to use
#         num_classes: number of classes for classification
#         pretrained: whether or not to get imagenet pretrained model
#
#     Return:
#         model: neural network object to be used
#     """
#     if name == 'resnet18':
#         model = models.resnet18
#         # model = resnet18
#     elif name == 'resnet34':
#         # model = models.resnet34
#         model = resnet34
#     elif name == 'resnet50':
#         # model = models.resnet50
#         model = resnet50
#     elif name == 'resnet101':
#         # model = models.resnet101
#         model = resnet101
#     elif name == 'resnet110':
#         model = resnet110
#     elif name == 'resnet152':
#         # model = models.resnet152
#         model = resnet152
#
#     # using models.resnetxxx
#     return model(num_classes=num_classes)

    # # using resnet.py
    # return model


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
        # optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
        return optimizer, scheduler
    elif name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4,
                              momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
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


def fast_collate(batch):
    """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)

    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8)
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        arr_shape = batch[0][0].shape
        if arr_shape[0] == 1:
            batch_shape = (3, arr_shape[1], arr_shape[2])
        else:
            batch_shape = batch[0][0].shape
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch_shape), dtype=torch.uint8)
        for i in range(batch_size):
            batch_arr = batch[i][0]
            if arr_shape[0] == 1:
                batch_arr = np.repeat(batch_arr, 3, axis=0)
            tensor[i] += torch.from_numpy(batch_arr)
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        arr_shape = batch[0][0].shape
        if arr_shape[0] == 1:
            batch_shape = (3, arr_shape[1], arr_shape[2])
        else:
            batch_shape = batch[0][0].shape
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch_shape), dtype=torch.float)
        for i in range(batch_size):
            batch_arr = batch[i][0]
            if arr_shape[0] == 1:
                batch_arr = batch_arr.repeat_interleave(3, 0)
            tensor[i].copy_(batch_arr.float())
        return tensor, targets
    elif isinstance(batch[0][0], PIL.Image.Image):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        b_shape = np.array(batch[0][0].convert('RGB')).shape
        torch_shape = (b_shape[2], b_shape[0], b_shape[1])
        tensor = torch.zeros((batch_size, *torch_shape), dtype=torch.uint8)
        for i in range(batch_size):
            #            tensor[i] += torch.from_numpy(np.array(batch[i][0]))
            x = np.array(batch[0][0].convert('RGB'))
            tensor[i] += torch.from_numpy(np.transpose(x, (2, 0, 1)))
        return tensor, targets

    else:
        assert False
