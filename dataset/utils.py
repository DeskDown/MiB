import torch
import numpy as np
from tqdm import trange


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in trange(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None,
                  overlap=True, opts=None, col_exemplars=False):
    # Filter images without any label in LABELS (using labels not reordered)

    # exemplars collection
    exemplars_idxs = None
    exampler_labels = set(([] if labels_old is None else labels_old) + labels)
    exampler_labels.discard(0)
    exampler_labels.discard(255)
    groups = {lab: [] for lab in exampler_labels}
    if col_exemplars:
        print("exemplars will be collected for following labels.\n{}"
                .format(exampler_labels))

    idxs = []

    # use all the data in offline settings
    if opts is not None and opts.task == 'offline':
        return [i for i in range(len(dataset))], exemplars_idxs

    # Incremental settings
    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        def fil(c): return any(x in labels for x in cls)
    else:
        def fil(c): return any(x in labels for x in cls) and all(
            x in labels_cum for x in c)

    for i in trange(len(dataset)):
        target = np.array(dataset[i][1])
        cls = np.unique(target)
        if fil(cls):
            idxs.append(i)
        if col_exemplars:
            update(i, cls, exampler_labels, groups, opts.exemplars_size)
    
    if col_exemplars:
        exemplars_idxs = select_exemplars(groups, opts.exemplars_size)

    return idxs, exemplars_idxs


def update(i, cls, labels_cum, groups, exemplars_size):
    for c in cls:
        if c in labels_cum and len(groups[c]) < exemplars_size:
            groups[c].append(i)
            continue


def select_exemplars(groups, exemplars_size):
    final_list = [x for g in groups for x in groups[g]]
    return final_list

class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices of new classes in the whole set selected for subset
        ex_indices (sequence): Indices of exemplars
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
        exemplars_transform (callable): mask new classes from exemplars
    """

    def __init__(self, dataset, indices, ex_indices, transform=None, target_transform=None, exemplars_transform = None):
        self.dataset = dataset
        self.indices = indices + ex_indices if ex_indices is not None else indices
        self.new_classes_idxs = set(indices)
        self.exemplars_idxs = set(ex_indices)
        self.transform = transform
        self.target_transform = target_transform
        self.exemplars_transform = exemplars_transform


    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        # Remove future labels from target
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Mask new classes labels from exemplars
        if exemplars_transform is not None and idx not in self.new_classes_idxs:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.indices)


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """

    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(
            lambda x: x if x in self.labels else self.value))

        return sample
