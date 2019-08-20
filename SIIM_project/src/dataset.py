import cv2
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
from youtrain.factory import DataFactory
from transforms import test_transform, mix_transform
import pydicom

from skimage import exposure

import torchvision
import copy

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

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
        image = exposure.equalize_adapthist(image)  # contrast correction
        image = ((image * 255)).clip(0, 255).astype(np.uint8)
        # RLE_mask = self.csv_file.loc[self.csv_file['ImageId'] == name][" EncodedPixels"].values[0]
        # if RLE_mask.strip() != str(-1):
        #     rle_mask = rle2mask(RLE_mask[1:], 1024, 1024).T
        # else:
        #     rle_mask = np.zeros((1024, 1024))

        # RLE_mask = self.csv_file.loc[self.csv_file['ImageId'] == name]["EncodedPixels"].values[0]
        # rle_mask = run_length_decode(RLE_mask)
        #^^ new_version

        # dilate
        # img = cv2.imread(rle_mask, cv2.IMREAD_GRAYSCALE)
        # _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3, 3), np.uint8)
        # rle_mask = cv2.dilate(mask, kernel, iterations=2)



        # dict_trasnformns = self.transform(image=image, mask=rle_mask)
        # image = dict_trasnformns['image']
        # rle_mask = dict_trasnformns['mask']
        # return {"image": torchvision.transforms.ToTensor()(image), "mask": torchvision.transforms.ToTensor()(rle_mask)}
#SEGMENTATION
        # dict_transfors = self.transform(image=image[:,:,None], mask=rle_mask[:,:,None])
        # image = dict_transfors['image'] #.permute(2,0,1)
        # rle_mask = dict_transfors['mask'] #.permute(2, 0, 1)
# CLASSIFICATION

        dict_trasnforms = self.transform(image=image[:,:])#, mask=rle_mask[:,:])
        image = dict_trasnforms['image']
        # mask = dict_trasnforms['mask']
        label = self.csv_file['label'].values[index]
        image = image[None,:,:]
        final_image_EfficientNet = np.concatenate((image, image, image), axis=0)
        return {"image":final_image_EfficientNet, "mask":label}

    def __len__(self):
        return len(self.csv_file)


class TestDataset(BaseDataset):
    def __init__(self, path, image_csv, transform):
        super().__init__(image_csv, transform)
        self.path = path
        self.folds = image_csv
        #self.csv_file = image_csv
        self.csv_file =image_csv


    def __getitem__(self, index):
        name = self.csv_file.iloc[index].ImageId
        # image = pydicom.dcmread(os.path.join(self.path, name + '.dcm')).pixel_array
        # image = self.transform(image=image[:, :])['image']
        # image = image[None, :, :]
        # final_image_EfficientNet = np.concatenate((image, image, image), axis=0)
        # return final_image_EfficientNet
        image = pydicom.dcmread(os.path.join(self.path, name + '.dcm')).pixel_array
        image = exposure.equalize_adapthist(image)  # contrast correction
        image = ((image * 255)).clip(0, 255).astype(np.uint8)
        dict_trasnforms = self.transform(image=image[:, :])
        image = dict_trasnforms['image']
        image = image[None, :, :]
        # final_image_EfficientNet = np.concatenate((image, image, image), axis=0)
        return image

    def __len__(self):
        return len(self.csv_file)


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
