from random import sample

from pdb import set_trace as trace


def map_labels(samples, excluded_labels, included_labels):
    """
    :param samples - list of x,y pairs. For each pair, x is an array whereas y
is an integer value representing the labels of x
    :param excluded_labels - List of labels to be removed
    :param included_labels - List of labels for which the corresponding samples
may remain
     
    :returns list of labels with excluded labels removed but the remaining
labels are mapped in order to values between 0 and 5
    """
    x_samples, labels_without_excluded = get_samples_without_excluded(samples, excluded_labels)
    new_labels = []
    for y in labels_without_excluded:
        new_label = included_labels.index(y)
        new_labels.append(new_label)

    return new_labels


def map_labels_only(labels, excluded_labels, included_labels):
    """
    :param samples - 
    :param excluded_labels - 
    :param included_labels - 
     
    :returns
    """
    new_labels = []
    for y in labels:
        if y not in excluded_labels:
            new_labels.append(y)
    mapped_labels = []
    for new_y in new_labels:
        mapped_labels.append(included_labels.index(new_y))
    return mapped_labels


def map_labels_only_idx(labels, excluded_labels, included_labels):
    """
    :param labels - 
    :param excluded_labels - 
    :param included_labels - 
     
    :returns
    """
    new_labels = []
    idxs = []
    for idx, y in enumerate(labels):
        if y not in excluded_labels:
            new_labels.append(y)
            idxs.append(idx)
        else:
            new_labels.append(-1)
    mapped_labels = []
    for new_y in new_labels:
        if new_y == -1:
            mapped_labels.append(new_y)
        else:
            mapped_labels.append(included_labels.index(new_y))
    return mapped_labels, idxs


# The following three functions are new
def get_labels(dataset):
    """
     
    :returns
    """
    if hasattr(dataset, 'targets'):
        return dataset.targets
    else:
        return dataset.labels


# def get_train_dataset(dataset, transform, root, target):
#     # trace()
#     if target == "cifar10" or target == "mnist":
#         return dataset(root=root, train=True, transform=transform, download=True)
#     else:
#         return dataset(root=root, split='train', transform=transform, download=True)
#
#
# def get_test_dataset(dataset, transform, root, target):
#     if target == "cifar10" or target == "mnist":
#         return dataset(root=root, train=False, transform=transform, download=True)
#     else:
#         return dataset(root=root, split='test', transform=transform, download=True)


def get_idxs_without_excluded(labels, excluded_labels):
    """
    :param samples - 
    :param excluded_labels - 
     
    :returns
    """

    idxs = []
    for idx, y in enumerate(labels):
        if y not in excluded_labels:
            idxs.append(idx)
    return idxs


def get_samples_without_excluded(samples, excluded_labels):
    """
    :param samples - 
    :param excluded_labels - 
     
    :returns
    """
    #    labels_without_excluded = [x for x in labels if x not in excluded_labels]
    labels_without_excluded = []
    x_samples = []

    for x, y in samples:
        if y not in excluded_labels:
            labels_without_excluded.append(y)
            x_samples.append(x)

    return x_samples, labels_without_excluded


def select_random_labels(all_labels, num_labels):
    """
    :param all_labels -  list of all classes
    :param num_labels -  number of labels to sample from all_labels
     
    :returns both the included and randomly removed labels 
    """
    sampled_labels = sample(all_labels, num_labels)
    remaining_labels = []
    sampled_labels.sort()
    for y in all_labels:
        if y not in sampled_labels:
            remaining_labels.append(y)

    return remaining_labels, sampled_labels


def select_fixed_labels(all_labels, excluded_labels):
    """
    :param all_labels -  list of all classes
    :param num_labels -  number of labels to sample from all_labels
     
    :returns both the included and randomly removed labels 
    """
    sampled_labels = excluded_labels
    remaining_labels = [];
    sampled_labels.sort()
    for y in all_labels:
        if y not in sampled_labels:
            remaining_labels.append(y)

    return remaining_labels, sampled_labels


def get_string_list(mylist):
    """
    :param mylist - 
     
    :returns
    """
    ms = ''
    for l in mylist:
        ms = ms + str(l)
    return ms
