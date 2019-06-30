import torch
import torchvision
import pandas
from torch.utils.data import Dataset, DataLoader


class LoadData(Dataset):
    def __init__(self, csv_file, transforms):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """

        :return: len
        """