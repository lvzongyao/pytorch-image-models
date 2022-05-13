import numpy as np

import scipy.spatial.distance as spd

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from pdb import set_trace as trace
import torchvision

from sklearn.metrics import brier_score_loss
# from openmax import compute_openmax
# import torchattacks
# from JackKnife import *
from map_labels import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_excluded_labels = list(range(6, 10))  # [6,7,8,9]
test_excluded_labels = list(range(6))  # [0,1,2,3,4,5]

criterion = nn.CrossEntropyLoss()

transform_resize = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(96),
    transforms.ToTensor(),
    # transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
])


def get_train_dataset(dataset, transform, root, target):
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=True, transform=transform, download=False)
    else:
        return dataset(root=root, split='train', transform=transform, download=False)


def get_test_dataset(dataset, transform, root, target):
    if target == "cifar10" or target == "mnist":
        return dataset(root=root, train=False, transform=transform, download=False)
    else:
        return dataset(root=root, split='test', transform=transform, download=False)


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

    # transform = transform_train
    transform = transform_resize
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
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn = fast_collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn = fast_collate)

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

    # transform = transform_test
    transform = transform_resize
    shuffle = False
    batch_size = 128

    dataset = get_test_dataset(dataset, transform, root, target)

    #    test_idxs = get_idxs_without_excluded(dataset.targets, train_excluded_labels)
    #    test_modified = torch.utils.data.Subset(dataset, test_idxs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn= fast_collate)

    return loader


def reliability_diagrams(probs, labels, mode, experiment):
    """ Draw reliability diagrams from classifier output. Save generated
        figure on the path to fname.

    Args:
        probs: output probability estimation from classifier
        labels: correct label list (list of integers)
        mode: file path to save diagram figure
    """
    preds = np.argmax(probs, axis=1)
    # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
    confidences = probs.max(axis=1)
    bins = np.linspace(0, 1, num=10, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1

    acc_list = []
    for i in range(len(bins)):
        acc = 0
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)

        if bin_size == 0:
            acc_list.append(0)
        else:
            acc_list.append(np.sum(preds[bin_idx] == labels[bin_idx]) / bin_size)

    x_list = [0.1 * i + 0.05 for i in range(10)]
    legend = ['Outputs', 'Gap']

    plt.figure(figsize=(8, 8))
    plt.bar(x=x_list, height=acc_list, width=0.1, color='b', alpha=0.9,
            edgecolor='k', linewidth=3)
    plt.bar(x=x_list, height=np.linspace(0.1, 1, num=10), width=0.1, color='r',
            alpha=0.2, edgecolor='r', linewidth=3)

    plt.xlabel('Confidence', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([0.2 * i for i in range(1, 6)], fontsize=20)
    plt.yticks([0.2 * i for i in range(1, 6)], fontsize=20)
    plt.legend(legend, loc=2, fontsize=25)
    plt.tight_layout()
    plt.grid()

    plt.savefig('./figures/{}.png'.format(mode + experiment))


"""
nnlabels = np.array(new_labels)
nnlabels[nnlabels == -1] = 6
np.mean(np.argmax(probs,axis =1) == nnlabels)
"""


# def evaluate_open_set(probs, labels, threshold=0.7):
def evaluate_open_set(probs, labels, threshold=0.7, is_openmax=False):
    """
     
    :returns
    """
    new_labels = np.array(labels.copy())
    new_labels[new_labels > 5] = 6

    if is_openmax:
        openmax_acc = np.mean(np.argmax(probs, axis=1) == new_labels)
        print('Threshold accuracy {} calibration: {} %'.format("openmax", 100 * openmax_acc))
        new_probs = probs.copy()
    if not is_openmax:
        prods = np.prod(1 - probs, axis=1)
        new_probs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)

    # labels_arr = np.zeros((probs.shape[0], probs.shape[1] + 1))
    # labels_arr[np.arange(new_labels.size), new_labels] = 1
    # prods = np.prod(1 - probs, axis=1)
    # new_probs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)

    highest_conf = np.max(new_probs, axis=1)
    predictions = np.argmax(new_probs, axis=1)
    new_predictions = []
    for pred, conf in zip(predictions, highest_conf):
        if conf > threshold:
            new_predictions.append(pred)
        else:
            new_predictions.append(6)
    open_acc = 100 * np.mean(new_predictions == new_labels)

    return open_acc

    # new_labels = []
    # for y in labels:
    #     if y > 5:
    #         new_labels.append(-1)
    #     else:
    #         new_labels.append(y)
    # nnlabels = np.array(new_labels)
    # nnlabels[nnlabels == -1] = 6
    # openmax_acc = np.mean(np.argmax(probs, axis=1) == nnlabels)
    #
    # print('Threshold Accuracy {} calibration: {}'.format("openmax", openmax_acc))
    # confidences = probs.max(axis=1)
    # preds = np.argmax(probs, axis=1)
    # new_preds = []
    # for co, pr in zip(confidences, preds):
    #     if co < threshold:
    #         new_preds.append(-1)
    #     else:
    #         new_preds.append(pr)
    # new_preds = np.array(new_preds)
    # new_labels = np.array(new_labels)
    # return 100 * np.mean(new_preds == new_labels)


def evaluate_brier(probs, labels, threshold):
    """
     
    :returns
    """
    confidences = []
    max_confs = np.max(probs, axis=1)
    ones_arr = np.ones(probs.shape[1])

    binary_labels = labels.copy()
    binary_labels[binary_labels < 6] = 1
    binary_labels[binary_labels > 5] = 1
    #    other_labels = labels.copy()
    #    other_labels[other_labels > 5] = 6
    prods = np.prod(1 - probs, axis=1)
    new_confs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)
    #
    #    for pr, mc in zip(probs, max_confs):
    #        if  mc > threshold:
    #            confidences.append(mc)
    #        else:
    #            print(f"mc is {mc } ")
    #            conf = np.prod(ones_arr - pr)
    #            print(f"conf is {conf } ")
    #            confidences.append( conf)

    br_confs = np.max(new_confs, axis=1)
    #    manual_brier =  np.mean(np.sum((np.expand_dims(br_confs,1) - np.expand_dims(binary_labels,1))**2, axis = 1))
    manual_brier = np.mean((np.expand_dims(br_confs, 1) - np.expand_dims(binary_labels, 1)) ** 2, axis=0)
    #    thres_brier_score = brier_score_loss(binary_labels, confidences)
    return manual_brier


#    return thres_brier_score
def evaluate_brier_other(probs, labels, threshold):
    """
     
    :returns
    """
    binary_labels = labels.copy()
    binary_labels[labels > 5] = 6
    #    binary_labels[labels < 6] = 1
    labels_arr = np.zeros((probs.shape[0], probs.shape[1] + 1))
    labels_arr[np.arange(binary_labels.size), binary_labels] = 1
    prods = np.prod(1 - probs, axis=1)
    new_confs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)
    manual_brier = np.mean(np.sum((new_confs - labels_arr) ** 2, axis=1))
    return manual_brier


def expected_calibration_error(probs, labels, threshold=0.5):
    """ Calculate expected calibration error from classifier output

    Args:
        probs: output probability estimation (confidences) from classifier
        labels: correct label list (list of integers)
        mode: calibration method that generated input confidences

    Returns:
        float: overall expected calibration error computed
    """
    other_labels = labels.copy()
    other_labels[other_labels > 5] = 6
    prods = np.prod(1 - probs, axis=1)
    new_probs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)
    num_data = len(labels)
    ece_score = 0

    # preds = np.argmax(probs, axis=1)
    # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
    # confidences = probs.max(axis=1)
    confidences = []

    max_confs = np.max(new_probs, axis=1)
    # ones_arr = np.ones(probs.shape[1])
    predicts = []
    for pr, mc in zip(new_probs, max_confs):
        if mc > threshold:
            confidences.append(mc)
            predicts.append(np.argmax(pr))
        else:
            confidences.append(pr[-1])
            predicts.append(6)

    bins = np.linspace(0, 1, num=15, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1

    for i in range(len(bins)):
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)
        if bin_size == 0:
            continue
        else:
            bin_acc = np.sum(np.array(predicts)[bin_idx] == other_labels[bin_idx]) / bin_size
            bin_conf = np.sum(np.array(confidences)[bin_idx]) / bin_size
            ece_score += np.abs(bin_acc - bin_conf) * bin_size

    ece_score /= num_data

    return ece_score


# old ece function
# def expected_calibration_error(probs, labels, mode, threshold=0.5):
#     """ Calculate expected calibration error from classifier output
#
#     Args:
#         probs: output probability estimation (confidences) from classifier
#         labels: correct label list (list of integers)
#         mode: calibration method that generated input confidences
#
#     Returns:
#         float: overall expected calibration error computed
#     """
#     num_data = len(labels)
#     ece_score = 0
#
#     preds = np.argmax(probs, axis=1)
#     # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
#     confidences = probs.max(axis=1)
#     bins = np.linspace(0, 1, num=15, endpoint=False)
#     idxs = np.digitize(confidences, bins) - 1
#
#     for i in range(len(bins)):
#         bin_idx = (idxs == i)
#         bin_size = np.sum(bin_idx)
#         if bin_size == 0:
#             continue
#         else:
#             bin_acc = np.sum(preds[bin_idx] == labels[bin_idx]) / bin_size
#             bin_conf = np.sum(confidences[bin_idx]) / bin_size
#             ece_score += np.abs(bin_acc - bin_conf) * bin_size
#
#     ece_score /= num_data
#     print('ECE {} calibration: {}'.format(mode, ece_score))
#
#     #    acc_j, acc_std = error_resampler(probs, labels, get_accuracy)
#     #    print('Accuracy {} calibration: {}'.format(mode,acc_j))
#     ##    thres_brier_score = evaluate_brier(probs, labels, threshold)
#     #    thres_br, thres_br_std = error_resampler(probs, labels, evaluate_brier, threshold = threshold)
#     #    print('Brier {} calibration: {}'.format(mode,thres_br))
#
#     osr_thresh_acc = evaluate_open_set(probs, labels, threshold)
#     print('Threshold Accuracy {} calibration: {}'.format(mode, osr_thresh_acc))
#     return ece_score


# def train(model, dataset, optimizer, num_epoch, device, scheduler=None):
#     model.train()
#     loss_val = 0.0
#
#     logits = []
#     targets = []
#
#     for epoch in range(num_epoch):
#         for i, (data, labels) in enumerate(dataset):
#             data, labels = data.to(device), labels.to(device)
#             optimizer.zero_grad()
#
#             outputs = model(data)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             loss_val += loss.item()
#             if epoch + 1 == num_epoch:
#                 targets += list(labels.detach().cpu().numpy())
#                 logits.append(outputs.detach().cpu().numpy())
#
#         if scheduler:
#             scheduler.step()
#
#         print('Epoch {}, Step {} loss: {:.5f}'.format(epoch, i+1,
#               loss_val/(i+1)))
#         loss_val = 0.0
#
#     logits = np.concatenate(logits, axis=0)
#     return logits, np.array(targets)


# calculate accuracy
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


# def train(model, train_iter, test_iter, optimizer, num_epochs, device, scheduler=None):
#     net = model.to(device)
#     print("training on ", device)
#     loss = torch.nn.CrossEntropyLoss()
#     batch_count = 0
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         # save and reload model
#         torch.save(net.state_dict(), 'resnet18_cifar10.pt')
#         net = model.to(device)
#         net.load_state_dict(torch.load('resnet18_cifar10.pt'))
#
#         test_acc = evaluate_accuracy(test_iter, net)
#
#         scheduler.step()
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train(model, train_iter, test_iter, optimizer, num_epochs, device, scheduler=None, model_name=None):
    net = model.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        #         print('y.shape:', y.shape)
        #         print('n:', n)
        #         print('batch_count: ', batch_count)
        torch.save(net.state_dict(), model_name + '.pt')
        net = model.to(device)
        net.load_state_dict(torch.load(model_name + '.pt'))
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_adversarial(model, train_iter, test_iter, optimizer, num_epochs, device, scheduler=None, steps=4,
                      model_name=None):
    atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=steps)
    net = model.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            new_X = atk(X, y).cuda()
            y_hat = net(new_X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        #         print('y.shape:', y.shape)
        #         print('n:', n)
        #         print('batch_count: ', batch_count)
        torch.save(net.state_dict(), model_name + '_adv_steps_' + str(args.steps) + '.pt')
        net = model.to(device)
        net.load_state_dict(torch.load(model_name + '_adv_steps_' + str(args.steps) + '.pt'))
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def test_openmax(model, dataset, device, mean_filename, distance_filename):
    model.eval()

    logits = []
    targets = []
    correct = 0
    total = 0

    softmax = nn.Softmax(dim=1)

    softmaxes = []
    with torch.no_grad():
        for data, labels in dataset:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            softmaxes.append(softmax(outputs).detach().cpu())
            _, predicted = torch.max(outputs, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()

            targets += list(labels.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy())

    print("Accuracy of the network : {} %%".format(100 * correct / total))

    logits = np.concatenate(logits, axis=0)
    softmaxes = np.concatenate(softmaxes, axis=0)
    new_logits = []
    for lo, so in zip(logits, softmaxes):
        activation = {}
        activation['scores'] = torch.Tensor(so).unsqueeze(0)
        activation['activations'] = torch.Tensor(lo).unsqueeze(0)
        _, openmax_vector, openmax_fc8 = compute_openmax(model, activation, mean_filename, distance_filename,
                                                         distance_type='eucos')
        new_logits.append(openmax_fc8.squeeze())
    return np.array(new_logits), np.array(targets)


def test(model, dataset, device):
    model.eval()

    logits = []
    targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataset:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()

            targets += list(labels.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy())

    print("Accuracy of the network : {} %".format(100 * correct / total))

    logits = np.concatenate(logits, axis=0)
    return logits, np.array(targets)


def get_indices(index_bool):
    all_correct_numeric_index = []
    for i in range(len(index_bool)):
        # print(i)
        if index_bool[i] == True:
            all_correct_numeric_index.append(i)
        else:
            i = i + 1
    return all_correct_numeric_index


def compute_distances(MAV, feature):
    eu_dist, cos_dist, eucos_dist = [], [], []
    for feat in feature:
        eu_dist += [spd.euclidean(MAV, feat)]
        cos_dist += [spd.cosine(MAV, feat)]
        eucos_dist += [spd.euclidean(MAV, feat) / 200. + spd.cosine(
            MAV, feat)]
    distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return distances


def save_activations(net, mean_filename, distances_filename, trainloader_no_shuffle, dataset):
    """
     
    :returns filenames of activations (means and distances)
    """
    ###########################################################################
    ### collect activations 
    ###########################################################################

    pred = []
    with torch.no_grad():
        for x, y in trainloader_no_shuffle:
            pred.append(net(x.to(device)).argmax(dim=1))
    pred = torch.cat(pred).cpu()

    images = torch.Tensor(dataset.data)
    labels = torch.Tensor(get_labels(dataset))
    correct_train_images = images[labels == pred]
    correct_train_labels = labels[labels == pred]
    classes = torch.unique(correct_train_labels)

    indices = (labels == pred)
    correct_numeric_index = get_indices(indices)
    correct_train_data = torch.utils.data.Subset(dataset, correct_numeric_index)
    subsets = {}
    classes = torch.unique(correct_train_labels)  # tensor([0., 1., 2., 3., 4., 5.])
    for thisclass in classes:
        print(int(thisclass), end=' ')
        classbools = (correct_train_labels == thisclass)
        correct_numeric_index = get_indices(classbools)
        subsets[int(thisclass)] = torch.utils.data.Subset(correct_train_data, correct_numeric_index)
    count = 0
    activations = []
    with torch.no_grad():
        #     for x,y in mnist_train:
        #         ans = net(x.to(device).unsqueeze())
        #         if count > 6: break
        #         count += 1
        count = 0
        print(len(subsets))
        for i in range(len(subsets)):
            newsub = DataLoader(subsets[i], batch_size=64, shuffle=True)
            activation_for_subset = []
            for x, y in newsub:
                activation = net(x.to(device).float())
                activation_for_subset += [activation]
            activation_for_subset_cat = torch.cat(activation_for_subset).cpu()
            activations.append(np.array(activation_for_subset_cat))
            count += 1

    MAVS = []
    for act in activations:
        MAVS.append(np.mean(act, 0))
    mean_distances = []
    for i in range(len(MAVS)):
        mean_distances.append(compute_distances(MAVS[i], activations[i]))
    np.save(distances_filename, mean_distances)
    np.save(mean_filename, MAVS)


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
        else: batch_shape = batch[0][0].shape
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch_shape), dtype=torch.uint8)
        for i in range(batch_size):
            batch_arr = batch[i][0]
            if arr_shape[0] == 1:
                batch_arr = np.repeat(batch_arr, 3, axis = 0) 
            tensor[i] += torch.from_numpy(batch_arr)
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        arr_shape = batch[0][0].shape
        if arr_shape[0] == 1: 
            batch_shape = (3, arr_shape[1], arr_shape[2])
        else: batch_shape = batch[0][0].shape
        print(f"arr_shape is {arr_shape } ")
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch_shape), dtype=torch.uint8)
        for i in range(batch_size):
            batch_arr = batch[i][0]
            if arr_shape[0] == 1:
                batch_arr = batch_arr.repeat_interleave(3,0)
            tensor[i].copy_(batch_arr)
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
            tensor[i] += torch.from_numpy(np.transpose(x, (2,0,1)))
        return tensor, targets
         
    else:
        assert False
