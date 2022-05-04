import os

import torch
import numpy as np
from scipy.special import softmax

from utils_partitioned import *
# from train_partitioned_403 import *
# from train_partitioned_410 import *
from train_partitioned import *

from pdb import set_trace as trace

torch.manual_seed(0)

args = argparser()
# data_dir = os.path.join('/data/public_data', args.dataset.upper())
data_dir = os.path.join('E:/Datasets', args.dataset.upper())
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_cuda)
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

# calibration_list = ['histogram_binning',
#                     'matrix_scaling',
#                     'vector_scaling',
#                     'temperature_scaling']
calibration_list = ['temperature_scaling']

is_openmax = False


def main():
    if args.dataset == 'cifar10':
        num_classes = 6
        data_name = 'CIFAR'
        num_channels = 3
    elif args.dataset == 'cifar6':
        num_classes = 6
        data_name = 'CIFAR'
        num_channels = 3
    elif args.dataset == 'cifar100':
        num_classes = 100
        num_channels = 3
    elif args.dataset == 'svhn':
        num_classes = 6
        data_name = 'SVHN'
        num_channels = 3
    elif args.dataset == 'mnist':
        num_classes = 6
        data_name = 'MNIST'
        num_channels = 1

    model = get_model(args.model_type, num_classes=num_classes, num_channels=num_channels)
    model.to(device)
    optimizer, scheduler = get_optim(args.optimizer, model)
    tr_dataloader, val_dataloader = get_train_loader(args.dataset, data_dir)

    test_dataloader = get_test_loader(args.dataset, data_dir)

    # train the model from scratch or load trained model
    if not args.load_model:
        model_name = '{}_{}_{}'.format('open_' + args.model_type, args.dataset, str(args.num_epoch) + 'e')
        train(model, tr_dataloader, test_dataloader, optimizer, args.num_epoch, device, scheduler,
              model_name=model_name)
        # torch.save(model.state_dict(), model_name)
    else:
        model.load_state_dict(torch.load(args.load_model))

    print('Validation: ', end='')
    val_logits, val_labels = test(model, val_dataloader, device)

    # test_dataloader = get_test_loader(args.dataset, data_dir)
    print('OOD accuracy without the "unknown" label: ', end='')
    test_logits, test_labels = test(model, test_dataloader, device)
    test_probs = softmax(test_logits, axis=1)

    # evaluate confidence calibration of original model
    reliability_diagrams(test_probs, test_labels, mode='before')

    print('')
    threshold = 0
    # for i in range(0, 10):
    for i in range(0, 20):
        print("threshold is {:.2f}:".format(threshold))

        uncalibrated_osr_thresh_acc = evaluate_open_set(test_probs, test_labels, threshold, is_openmax=is_openmax)
        print('Threshold accuracy: {:>26.3f} %'.format(uncalibrated_osr_thresh_acc))

        # thres_brier_score = evaluate_brier(test_probs, test_labels, threshold)
        thres_brier_score = evaluate_brier_other(test_probs, test_labels, threshold)
        print('Brier before calibration: {0:>31}'.format(thres_brier_score))

        thres_ece = expected_calibration_error(test_probs, test_labels, threshold=threshold)
        print('ECE before calibration: {0:>34}'.format(thres_ece))

        print("calibrating...")
        for cali in calibration_list:
            calibrator = get_calibrator(val_logits, val_labels, mode=cali)
            confidences = calibrator(test_logits)
            reliability_diagrams(confidences, test_labels, mode=cali)
            # thres_cali_brier_score = evaluate_brier(confidences, test_labels, threshold=threshold)
            thres_cali_brier_score = evaluate_brier_other(confidences, test_labels, threshold=threshold)
            print('Brier {} calibration:  {}'.format(cali, thres_cali_brier_score))
            thres_cali_ece = expected_calibration_error(confidences, test_labels, threshold=threshold)
            print('ECE {} calibration:    {}'.format(cali, thres_cali_ece))
        # threshold += 0.1
        threshold += 0.05
        print('')


if __name__ == '__main__':
    main()
