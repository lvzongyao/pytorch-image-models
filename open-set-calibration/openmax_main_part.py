import os

import torch
import numpy as np
from scipy.special import softmax

from train_partitioned import *
from timm.models import create_model
# from pdb import set_trace as trace

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
args = argparser()
# data_dir = os.path.join('/data/public_data', args.dataset.upper())
data_dir = os.path.join('E:/Datasets', args.dataset.upper())
if args.dataset == 'mnist':
    data_dir = os.path.join('E:/Datasets')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_cuda)

# calibration_list = ['histogram_binning',
#                     'matrix_scaling',
#                     'vector_scaling',
#                     'temperature_scaling']
calibration_list = ['temperature_scaling']

is_openmax = True


def main():
    if args.dataset == 'cifar10':
        num_classes = 6
    elif args.dataset == 'cifar6':
        num_classes = 6
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'svhn':
        num_classes = 6
    elif args.dataset == 'mnist':
        num_classes = 6


    # model = get_model(args.model_type, num_classes=num_classes)
    # model = get_model(args.model_type, num_classes=num_classes, num_channels=3)
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

    test_dataloader = get_test_loader(args.dataset, data_dir)

    # train the model from scratch or load trained model
    if not args.load_model:
        model_name = '{}_{}.ckpt'.format(args.model_type, args.dataset)
        train(model, tr_dataset, test_dataloader, optimizer, args.num_epoch, device, scheduler)
        # torch.save(model.state_dict(), model_name)
    else:
        model.load_state_dict(torch.load(args.load_model))

    print('OOD accuracy without the "unknown" label: ', end='')
    test_logits, test_labels = test(model, test_dataloader, device)

    print('testing openmax...')
    m_filename, d_filename = "openmax_edit/mean_acts", "openmax_edit/dist_acts"
    ###########################################################################
    ### uncomment next few lines when creating new activations  
    ###########################################################################
    target, root = args.dataset, data_dir
    save_activations(model, m_filename, d_filename, target, root)

    test_logits, test_labels = test_openmax(model, test_dataloader, device , m_filename + ".npy",d_filename + ".npy")
    np.save("openmax_edit/test_logits", test_logits)
    np.save("openmax_edit/test_labels", test_labels)

    test_logits, test_labels = np.load("openmax_edit/test_logits.npy"), np.load("openmax_edit/test_labels.npy")
    test_probs = softmax(test_logits, axis=1)

    # evaluate confidence calibration of original model
    reliability_diagrams(test_probs, test_labels, mode='before')

    val_logits, val_labels = test_openmax(model, val_dataset, device , m_filename + ".npy",d_filename + ".npy")
    np.save("openmax_edit/val_logits", val_logits)
    np.save("openmax_edit/val_labels", val_labels)
    # val_logits, val_labels = np.load("openmax_edit/val_logits.npy"), np.load("openmax_edit/val_labels.npy")

    print('')
    threshold = 0
    for i in range(0, 20):
        print("threshold is {:.2f}:".format(threshold))

        uncalibrated_osr_thresh_acc = evaluate_open_set(test_probs, test_labels, threshold, is_openmax=is_openmax)
        print('Threshold accuracy: {:>26.3f} %'.format(uncalibrated_osr_thresh_acc))

        # thres_brier_score = evaluate_brier(test_probs, test_labels, threshold)
        thres_brier_score = evaluate_brier_other(test_probs, test_labels, threshold)
        print('Brier before calibration: {0:>33}'.format(thres_brier_score))

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
        threshold += 0.05
        print('')


if __name__ == '__main__':
    main()
