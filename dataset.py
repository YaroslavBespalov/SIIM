import torch
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import init_img_mask
import os

def get_path(idx):
    return os.


class LoadData(Dataset):
    def __init__(self, csv_file, transforms):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """

        :return: len
        """
        return len(self.csv_file)
    def __getitem__(self, item):
        """

        :param item: id of returned item
        :return:
        """
        image, mask, _ = init_img_mask()
        return {"image": , "mask": }