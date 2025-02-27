# import os, sys, pickle, glob
# import os.path as path
import sys
import argparse
import numpy as np
# import scipy.spatial.distance as spd
import scipy as sp
from scipy.io import loadmat

from openmax_edit.openmax_utils import compute_distance, getlabellist
from openmax_edit.evt_fitting import weibull_tailfitting, query_weibull
from pdb import set_trace as trace
try:
    # import utils.libmr    # noqa: F401
    import libmr
except ImportError:
    print("LibMR not installed or libmr.so not found")
    print("Install libmr: cd libMR/; ./compile.sh")
    sys.exit()

# ---------------------------------------------------------------------------------
# params and configuratoins
NCHANNELS = 1
NCLASSES = 6
ALPHA_RANK = 6
WEIBULL_TAIL_SIZE = 10


# ---------------------------------------------------------------------------------


def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    """ Convert the scores in probability value using openmax

    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree

    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class

    """
    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []  # noqa: F841
        for category in range(NCLASSES):
            # print (channel,category)
            # print ('openmax',openmax_fc8[channel, category])

            channel_scores += [sp.exp(openmax_fc8[channel, category])]
        # print ('CS',channel_scores)

        total_denominator = sp.sum(
            sp.exp(openmax_fc8[channel, :])) + sp.exp(
            sp.sum(openmax_score_u[channel, :]))
        # print (total_denominator)

        prob_scores += [channel_scores / total_denominator]
        # print (prob_scores)

        prob_unknowns += [
            sp.exp(sp.sum(openmax_score_u[channel, :])) / total_denominator]

    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns)

    scores = sp.mean(prob_scores, axis=0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    #    assert len(modified_scores) == 11
    return modified_scores


# ---------------------------------------------------------------------------------

# def recalibrate_scores(weibull_model, labellist, imgarr,
#                        layer='fc8', alpharank=6, distance_type='eucos'):
def recalibrate_scores(weibull_model, labellist, imgarr,
                       layer='activations', alpharank=6, distance_type='eucos'):
    """
    Given FC8 features for an image, list of weibull models for each class,
    re-calibrate scores

    Input:
    ---------------
    weibull_model : pre-computed weibull_model obtained
     from weibull_tailfitting() function
    labellist : ImageNet 2012 labellist
    imgarr : features for a particular image extracted using caffe architecture

    Output:
    ---------------
    openmax_probab: Probability values for a given class computed using OpenMax
    softmax_probab: Probability values for a given class computed using
     SoftMax (these were precomputed from caffe architecture. Function returns
     them for the sake of convenience)

    """
    imglayer = imgarr[layer]
    # ranked_list = imgarr['scores'].argsort().ravel()[::-1]
    _, ranked_list = imgarr['scores'].sort(descending=True)
    ranked_list = ranked_list.ravel()

    alpha_weights = [
        ((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = sp.zeros(10)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # print (imglayer)
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = imglayer[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        # count = 0
        for categoryid in range(NCLASSES):
            # get distance between current channel and mean vector
            category_weibull = query_weibull(
                labellist[categoryid],
                weibull_model, distance_type=distance_type)

            # print (
            #    category_weibull[0], category_weibull[1],category_weibull[2])

            channel_distance = compute_distance(
                channel_scores, channel, category_weibull[0],
                distance_type=distance_type)
            # print ('cd',channel_distance)
            # obtain w_score for the distance and compute probability of the
            # distance
            # being unknown wrt to mean training vector and channel distances
            # for # category and channel under consideration
            wscore = category_weibull[2][channel].w_score(channel_distance)
            # print ('wscore',wscore)
            # print (channel_scores)
            modified_fc8_score = channel_scores[categoryid] * (
                    1 - wscore * ranked_alpha[categoryid])
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [
                channel_scores[categoryid] - modified_fc8_score]

        # gather modified scores fc8 scores for each channel for the
        # given image
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    np.array([[np.sum(np.array(channel_scores)) - np.sum(openmax_fc8)]])
    new_act = np.concatenate([openmax_fc8, np.array([[np.sum(np.array(channel_scores)) - np.sum(openmax_fc8)]])], axis = 1)
    # print (openmax_fc8,openmax_score_u)
    # Pass the recalibrated fc8 scores for the image into openmax
    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = imgarr['scores'].ravel()
    return sp.asarray(openmax_probab), sp.asarray(softmax_probab), new_act


# ---------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    # Optional arguments.
    parser.add_argument(
        "--weibull_tailsize",
        type=int,
        default=WEIBULL_TAIL_SIZE,
        help="Tail size used for weibull fitting"
    )

    parser.add_argument(
        "--alpha_rank",
        type=int,
        default=ALPHA_RANK,
        help="Alpha rank to be used as a weight multiplier for top K scores"
    )

    parser.add_argument(
        "--distance",
        default='eucos',
        help="Type of distance to be used for calculating distance \
        between mean vector and query image \
        (eucos, cosine, euclidean)"
    )

    parser.add_argument(
        "--mean_files_path",
        default='data/mean_files/',
        help="Path to directory where mean activation vector (MAV) is saved."
    )

    parser.add_argument(
        "--synsetfname",
        default='synset_words_caffe_ILSVRC12.txt',
        help="Path to Synset filename from caffe website"
    )

    parser.add_argument(
        "--image_arrname",
        default='data/train_features/n01440764/n01440764_14280.JPEG.mat',
        help="Image Array name for which openmax scores are to be computed"
    )

    parser.add_argument(
        "--distance_path",
        default='data/mean_distance_files/',
        help="Path to directory where distances of training data \
        from Mean Activation Vector is saved"
    )

    args = parser.parse_args()

    distance_path = args.distance_path
    mean_path = args.mean_files_path
    # alpha_rank = args.alpha_rank
    # weibull_tailsize = args.weibull_tailsize
    synsetfname = args.synsetfname
    image_arrname = args.image_arrname

    labellist = getlabellist(synsetfname)
    weibull_model = weibull_tailfitting(mean_path, distance_path, labellist,
                                        tailsize=WEIBULL_TAIL_SIZE)

    print("Completed Weibull fitting on %s models" % len(weibull_model.keys()))
    imgarr = loadmat(image_arrname)
    openmax, softmax = recalibrate_scores(weibull_model, labellist, imgarr)
    print("Image ArrName: %s" % image_arrname)
    print("Softmax Scores ", softmax)
    print("Openmax Scores ", openmax)
    print(openmax.shape, softmax.shape)


if __name__ == "__main__":
    main()
