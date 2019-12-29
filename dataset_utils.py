import os
import glob
import shutil
import sys
from textwrap import indent

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class TinyImagenetDataset(Dataset):
    """
    Custom dataset implementation for Tiny Imagenet Dataset configuration. Please note that this way is not the
    preferred method over "alternativeSeperation" function. Using this class may introduce some overhead.

    """

    def __init__(self, path="data", part="train", transform=None):
        """
        Custom dataset implementation for Tiny Imagenet Dataset configuration.
        :param path: Root path of the dataset
        :param transform: Transformation to apply
        """
        self.transform = transform
        self.path = os.path.join(path, part)
        self.part = part

        if part == "val":
            val_df = pd.read_csv(os.path.join(self.path, "val_annotations.txt"), delimiter="\t",
                                 header=None, index_col=0)

            self.val_labels = val_df.to_dict()[1]

        class_labels = pd.read_csv(os.path.join(path, "words.txt"), delimiter="\t",
                                   header=None, index_col=0)

        self.class_labels = class_labels.to_dict()[1]

        self.image_dir_list = glob.glob(self.path + '/**/*.JPEG', recursive=True)

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, item):
        image_dir = self.image_dir_list[item]
        img = Image.open(image_dir)

        if self.transform is not None:
            img = self.transform(img)

        if self.part == "train":
            label = image_dir.split("/")[2]
        else:
            label = self.val_labels[image_dir.split("/")[-1]]

        return img, label


def alternativeSeperation(path="data"):
    """
    Preferred alternative method for feeding images. Enables PyTorch's ImageDataset utilization.This function simply
    separates the images into corresponding label folder. This method may help save some RAM and remove some overhead
    comparing to using TinyImagenetDataset.
    :param path: Root path of the dataset
    """
    path = os.path.join(path, "val")
    val_df = pd.read_csv(os.path.join(path, "val_annotations.txt"), delimiter="\t",
                         header=None, index_col=0)
    val_labels = val_df.to_dict()[1]

    for image, label in val_labels.items():
        label_path = os.path.join(path, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        shutil.move(os.path.join(os.path.join(path, "images"), image), label_path)

