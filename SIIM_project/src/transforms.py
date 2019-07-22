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


def pre_transform(**kwargs):
    return Compose([Resize(height=kwargs['pre_resize_size'], width=kwargs['pre_resize_size']),
                    ])

def post_transform(**kwargs):
    return Compose([
        # Normalize(
        #     mean=(0.485),
        #     std=(0.229)),
        ToTensor()
    ])


def mix_transform(**kwargs):
    return Compose([
        pre_transform(**kwargs),
        #Rotate(limit=10, interpolation=cv2.INTER_LINEAR),
       # IAAAdditiveGaussianNoise(p=0.25),
       # VerticalFlip(),
       HorizontalFlip(p=kwargs['mix_hflip']),
        RandomRotate90(),
        VerticalFlip(),
        OpticalDistortion(),
        GridDistortion(),
        ElasticTransform(),
        MedianBlur(),
      #  RandomGamma(),
      #  RandomRotate90(),
        post_transform(**kwargs)
    ])

def test_transform(**kwargs):
    return Compose([
        pre_transform(**kwargs),
        post_transform()]
    )
