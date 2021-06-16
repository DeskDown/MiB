import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .utils import Subset, filter_images, group_images

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 image_set='train',
                 is_aug=True,
                 transform=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = image_set
        base_dir = "PascalVOC12"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if is_aug and image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(
                mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(
            voc_root, x[1][1:])) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 opts=None):

        full_voc = VOCSegmentation(
            root, 'train' if train else 'val', is_aug=True, transform=None)

        idxs = None
        self.labels = []
        self.labels_old = []
        add_exemplars = False
        col_exemplars = False
        exemplars_idxs = None
        new_exemplars_idxs = None
        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)
            if opts.use_exemplars == False:
                assert not any(l in labels_old for l in labels),\
                    "labels and labels_old must be disjoint sets"

            # exemplars setup
            exemplars_path = os.path.join(os.path.dirname(
                idxs_path), f'exemplars_{opts.step-1}_{opts.exemplars_size}.npy')
            new_exemplars_path = os.path.join(os.path.dirname(
                idxs_path), f'exemplars_{opts.step}_{opts.exemplars_size}.npy')
            if train:
                if opts.use_exemplars and os.path.exists(exemplars_path):
                    add_exemplars = True
                if opts.col_exemplars and not os.path.exists(new_exemplars_path):
                    col_exemplars = True
                if opts.step > 0 and opts.use_exemplars:
                    if os.path.exists(exemplars_path):
                        exemplars_idxs = np.load(exemplars_path).tolist()
                    else:
                        raise f"exemplars not found: {exemplars_path}"


            # take index of images with at least one class in labels
            # and all classes in labels+labels_old+[0,255]
            # and take care of exemplars
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            if idxs_path is None or not os.path.exists(idxs_path) or col_exemplars:
                idxs, new_exemplars_idxs = filter_images(full_voc, labels, labels_old,
                                                         overlap=overlap, opts=opts,
                                                         col_exemplars=col_exemplars)
                if idxs_path is not None:  # and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))
                if new_exemplars_idxs is not None:
                    print("exemplars selected for next steps: {}"
                            .format(len(new_exemplars_idxs)))
                    np.save(new_exemplars_path, np.array(new_exemplars_idxs, dtype=int))

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old
            self.order = [0] + labels_old + labels

            if train and add_exemplars:
                print("Original train size:{} exemplars to be added:{}"
                      .format(len(idxs), len(exemplars_idxs)))
                idxs = idxs + exemplars_idxs*10
                self.labels = [0] + labels_old + labels

            if train:
                masking_value = 0
            else:
                masking_value = 255

            self.inverted_order = {label: self.order.index(
                label) for label in self.order}
            self.inverted_order[255] = masking_value

            reorder_transform = tv.transforms.Lambda(
                lambda t: t.apply_(lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:
                tmp_labels = self.labels + [255]
                target_transform = tv.transforms.Lambda(
                    lambda t: t.apply_(lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            # make the subset of the dataset
            self.dataset = Subset(full_voc, idxs, transform, target_transform)
        else:
            self.dataset = full_voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
