import os

import torch
import numpy as np
from scipy.special import softmax

from utils import *
from train import *

from timm.models import create_model

torch.manual_seed(0)

args = argparser()
# data_dir = os.path.join('/data/public_data', args.dataset.upper())
data_dir = os.path.join('E:/Datasets', args.dataset.upper())
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_cuda)

# calibration_list = ['histogram_binning',
#                     'matrix_scaling',
#                     'vector_scaling',
#                     'temperature_scaling']
calibration_list = ['temperature_scaling']


def main():
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'svhn':
        num_classes = 10
        data_name = 'SVHN'
        num_channels = 3
    elif args.dataset == 'mnist':
        num_classes = 10

    print('num of classes: ', num_classes)

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    # model = get_model(args.model_type, num_classes=num_classes)
    model = create_model(
        # args.model,
        args.model_type,
        num_classes=num_classes,

        # input_size=(1, 32, 32),

        # pretrained=args.pretrained,
        # num_classes=args.num_classes,
        # drop_rate=args.drop,
        # drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        # drop_path_rate=args.drop_path,
        # drop_block_rate=args.drop_block,
        # global_pool=args.gp,
        # bn_momentum=args.bn_momentum,
        # bn_eps=args.bn_eps,
        # scriptable=args.torchscript,
        # checkpoint_path=args.initial_checkpoint
    )

    model.to(device)
    optimizer, scheduler = get_optim(args.optimizer, model)
    tr_dataset, val_dataset = get_train_loader(args.dataset, data_dir)

    test_dataset = get_test_loader(args.dataset, data_dir)

    # train the model from scratch or load trained model
    if not args.load_model:
        model_name = '{}_{}'.format(args.model_type, args.dataset)
        train(model, tr_dataset, test_dataset, optimizer, args.num_epoch, device, scheduler)
        torch.save(model.state_dict(), model_name)
    else:
        model.load_state_dict(torch.load(args.load_model))

    print('Validation: ', end='')
    val_logits, val_labels = test(model, val_dataset, device)

    # test_dataset = get_test_loader(args.dataset, data_dir)
    print('Testing: ', end='')
    test_logits, test_labels = test(model, test_dataset, device)
    test_probs = softmax(test_logits, axis=1)

    # evaluate confidence calibration of original model
    reliability_diagrams(test_probs, test_labels, mode='before')
    brier_uncali = evaluate_closed_brier(test_probs, test_labels)
    print('Brier before calibration {}'.format(brier_uncali))
    ece_uncalibrated = expected_calibration_error(test_probs, test_labels)
    print('ECE before calibration: {}'.format(ece_uncalibrated))

    print('calibrating...')
    for cali in calibration_list:
        calibrator = get_calibrator(val_logits, val_labels, mode=cali)
        confidences = calibrator(test_logits)

        reliability_diagrams(confidences, test_labels, mode=cali)

        brier_cali = evaluate_closed_brier(confidences, test_labels)
        print('Closed Brier {} {}'.format(cali, brier_cali))

        ece_calibrated = expected_calibration_error(confidences, test_labels)
        print('ECE {} calibration: {}'.format(cali, ece_calibrated))

    '''brier score version'''  # wrong
    # threshold = 0
    # for i in range(0, 20):
    #     print(f"threshold is {threshold}:")
    #     thres_brier_score = evaluate_closed_brier(test_probs, test_labels, threshold)
    #     print('Brier {} without calibration: {}'.format("Uncalibrated", thres_brier_score))
    #     # uncalibrated_osr_thresh_acc = evaluate_open_set(test_probs, test_labels, threshold)
    #     # print('Threshold Accuracy {} without calibration: {}'.format("Uncalibrated", uncalibrated_osr_thresh_acc))
    #     for cali in calibration_list:
    #         print("calibrating...")
    #         calibrator = get_calibrator(val_logits, val_labels, mode=cali)
    #         confidences = calibrator(test_logits)
    #         reliability_diagrams(confidences, test_labels, mode=cali)
    #         expected_calibration_error(confidences, test_labels, mode=cali, threshold=threshold)
    #     # threshold += 0.1
    #     threshold += 0.05
    # print('\n')


if __name__ == '__main__':
    main()
