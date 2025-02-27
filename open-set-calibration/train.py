import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss

import time

from utils import *

criterion = nn.CrossEntropyLoss()

calibration_list = ['temperature_scaling']
is_openmax = False
args = argparser()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reliability_diagrams(probs, labels, mode):
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
    plt.savefig('./figures/{}.png'.format(mode))


def evaluate_closed_brier(probs, labels):
    # """
    #
    # :returns
    # """
    # confidences = []
    # max_confs = np.max(probs, axis=1)
    # ones_arr = np.ones(probs.shape[1])
    #
    # binary_labels = labels.copy()
    # binary_labels[binary_labels < 6] = 1
    # binary_labels[binary_labels > 5] = 1
    # #    other_labels = labels.copy()
    # #    other_labels[other_labels > 5] = 6
    # prods = np.prod(1 - probs, axis=1)
    # new_confs = np.concatenate([probs, np.expand_dims(prods, 1)], axis=1)
    # #
    # #    for pr, mc in zip(probs, max_confs):
    # #        if  mc > threshold:
    # #            confidences.append(mc)
    # #        else:
    # #            print(f"mc is {mc } ")
    # #            conf = np.prod(ones_arr - pr)
    # #            print(f"conf is {conf } ")
    # #            confidences.append( conf)
    #
    # br_confs = np.max(new_confs, axis=1)
    # #    manual_brier =  np.mean(np.sum((np.expand_dims(br_confs,1) - np.expand_dims(binary_labels,1))**2, axis = 1))
    # manual_brier = np.mean((np.expand_dims(br_confs, 1) - np.expand_dims(binary_labels, 1)) ** 2, axis=0)
    # #    thres_brier_score = brier_score_loss(binary_labels, confidences)
    # return manual_brier
    # #    return thres_brier_score

    # confidences = []
    # max_confs = np.max(probs, axis=1)
    # ones_arr = np.ones(probs.shape[1])
    #
    # binary_labels = labels.copy()
    # binary_labels[binary_labels < 6] = 1
    # binary_labels[binary_labels > 5] = 0
    # # other_labels = labels.copy()
    # # other_labels[other_labels > 5] = 6
    # # trace()
    # # prods = np.prod(1 - probs, axis = 1)
    # # new_confs = np.concatenate([probs, np.expand_dims(prods, 1)], axis = 1)
    #
    # for pr, mc in zip(probs, max_confs):
    #     if mc > threshold:
    #         confidences.append(mc)
    #     else:
    #         # print(f"mc is {mc } ")
    #         conf = np.prod(ones_arr - pr)
    #         # print(f"conf is {conf } ")
    #         confidences.append(1 - conf)

    # thres_brier_score = brier_score_loss(binary_labels, confidences)
    labels_arr = np.zeros(probs.shape)
    labels_arr[np.arange(labels.size), labels] = 1
    manual_brier = np.mean(np.sum((probs - labels_arr) ** 2, axis=1))
    return manual_brier
    # return thres_brier_score


# def expected_calibration_error(probs, labels, mode):
def expected_calibration_error(probs, labels):
    """ Calculate expected calibration error from classifier output

    Args:
        probs: output probability estimation (confidences) from classifier
        labels: correct label list (list of integers)
        mode: calibration method that generated input confidences
        threshold:

    Returns:
        float: overall expected calibration error computed
    """
    num_data = len(labels)
    ece_score = 0

    preds = np.argmax(probs, axis=1)
    # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
    confidences = probs.max(axis=1)
    bins = np.linspace(0, 1, num=15, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1

    for i in range(len(bins)):
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)
        if bin_size == 0:
            continue
        else:
            bin_acc = np.sum(preds[bin_idx] == labels[bin_idx]) / bin_size
            bin_conf = np.sum(confidences[bin_idx]) / bin_size
            ece_score += np.abs(bin_acc - bin_conf) * bin_size

    ece_score /= num_data

    return ece_score


'''original training code'''
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


'''custom training code'''


# calculate accuracy
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # use device of net if not device specified 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # evaluation mode, dropout off
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            n += y.shape[0]
    return acc_sum / n


def train(model, train_iter, valid_iter, test_iter, optimizer, num_epochs, device, scheduler=None, model_name=None):
    net = model.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    best_acc = 0
    best_count = 0
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

        # save and reload model
        # torch.save(net.state_dict(), './checkpoints/' + model_name + '.pt')
        # net = model.to(device)
        # net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

        test_acc = evaluate_accuracy(test_iter, net)

        if test_acc > best_acc:
            best_count = 0
            best_acc = test_acc
            torch.save(net.state_dict(), './checkpoints/' + model_name + '.pt')

        net = model.to(device)
        net.load_state_dict(torch.load('./checkpoints/' + model_name + '.pt'))

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        val_logits, val_labels = test(model, valid_iter, device)
        test_logits, test_labels = test(model, test_iter, device)
        test_probs = softmax(test_logits, axis=1)
        reliability_diagrams(test_probs, test_labels, mode='before')

        brier_uncali = evaluate_closed_brier(test_probs, test_labels)
        print('Brier before calibration {}'.format(brier_uncali))
        ece_uncalibrated = expected_calibration_error(test_probs, test_labels)
        print('ECE before calibration: {}'.format(ece_uncalibrated))

        for cali in calibration_list:
            calibrator = get_calibrator(val_logits, val_labels, mode=cali)
            confidences = calibrator(test_logits)

            reliability_diagrams(confidences, test_labels, mode=cali)

            brier_cali = evaluate_closed_brier(confidences, test_labels)
            print('Closed Brier {} {}'.format(cali, brier_cali))

            ece_calibrated = expected_calibration_error(confidences, test_labels)
            print('ECE {} calibration: {}'.format(cali, ece_calibrated))

        # if test_acc <= best_acc:
        #     best_count +=1
        #     if best_count > 2: break


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

    print("Accuracy of the network: {} %".format(100 * correct / total))

    logits = np.concatenate(logits, axis=0)
    return logits, np.array(targets)
