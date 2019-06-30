import cv2
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from youtrain.factory import DataFactory
from transforms import test_transform, mix_transform
import pydicom
import torchvision
import copy



def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


class BaseDataset(Dataset):
    def __init__(self, folds, transform):
        self.csv = folds
        self.transform = transform

    #        print(len(self.ids))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, path, folds, transform=None):
        super().__init__(folds, transform)
        self.path = path
        self.folds = folds
        self.csv_file = folds
        self.transform = transform
    def __getitem__(self, index):
        name = self.csv_file.iloc[index].ImageId

        image = pydicom.dcmread(os.path.join(self.path, name + '.dcm')).pixel_array
        RLE_mask = self.csv_file.loc[self.csv_file['ImageId'] == name][" EncodedPixels"].values[-1]
        if RLE_mask.strip() != str(-1):
            rle_mask = rle2mask(RLE_mask[1:], 1024, 1024).T
        else:
            rle_mask = np.zeros((1024, 1024))

        # if self.transform is not None:
        #     image = self.transform(image)
        #     rle_mask = self.transform(rle_mask)

        image = torchvision.transforms.ToTensor()(image)
        rle_mask = torchvision.transforms.ToTensor()(rle_mask)

        return {"image": image, "mask": rle_mask}

    def __len__(self):
        return len(self.csv_file)


class TestDataset(BaseDataset):
    def __init__(self, image_dir, ids, transform):
        super().__init__(image_dir, ids, transform)
        self.transform = transform
        self.ids = ids
        self.image_dir = image_dir

    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.imread(os.path.join(self.image_dir, name))
        return self.transform(image=image)['image']


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None

    @property
    def data_path(self):
        return Path(self.paths['path'])

    def make_transform(self, stage, is_train=False):
        if is_train:
            if stage['augmentation'] == 'mix_transform':
                transform = mix_transform(**self.params['augmentation_params'])
            else:
                raise KeyError('augmentation does not found')
        else:
            transform = test_transform(**self.params['augmentation_params'])
        return transform

    def make_dataset(self, stage, is_train):
        transform = self.make_transform(stage, is_train)
        folds = self.train_ids if is_train else self.val_ids
        return TrainDataset(
            path=str(self.data_path),
            #            mask_dir=self.data_path / self.paths['train_masks'],
            folds=folds,
            transform=transform)

    def make_loader(self, stage, is_train=False):
        dataset = self.make_dataset(stage, is_train)
        return DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.params['num_workers'],
            pin_memory=torch.cuda.is_available(),
        )

    @property
    def folds(self):
        if self._folds is None:
            self._folds = pd.read_csv(self.data_path / self.paths['folds'])
        return self._folds

    @property
    def train_ids(self):
        return self.folds.loc[self.folds['fold'] != self.fold]

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold]
