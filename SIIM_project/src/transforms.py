from albumentations import *
from albumentations.torch import ToTensor
import numpy as np
import cv2
import random
import numpy as np
# from composition import Compose, OneOf, GrayscaleOrColor
# import functional as F
from imgaug import augmenters as iaa
from scipy.ndimage import label


def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)

def post_transform():
    return Compose([
        Normalize(
            mean=(0.485),
            std=(0.229)),
        ToTensor()
    ])


def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        # VerticalFlip(),
        # HorizontalFlip(),
        # CLAHE(),
        # RandomRotate90(),
        # CLAHE(p=0.3),
        HorizontalFlip(p=0.5),
        OneOf([
            RandomRotate90(),
            # ShiftScaleRotate(
            #     shift_limit=0,  # no resizing
            #     scale_limit=0.1,
            #     rotate_limit=10,  # rotate
            #     border_mode=cv2.BORDER_CONSTANT
            # )
        ], p=0.3),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            CLAHE(),
            RandomBrightness(),
        ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
      ## RandomSizedCrop(min_max_height=(128, 256), height=256, width=256, p=0.5),
        post_transform()
    ])

def test_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()]
    )
