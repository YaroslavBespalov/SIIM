import numpy as np
import pandas as pd
import os

import argparse
from pathlib import Path

import cv2
import pydoc
import torch


from tqdm import tqdm
from dataset import TestDataset
#from inference import PytorchInference
from transforms import test_transform
from torch.utils.data import DataLoader
from youtrain.utils import set_global_seeds, get_config, get_last_save
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def mask_to_rle(img, width=1024, height=1024):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    return " " + " ".join(rle)


class PytorchInference:
    def __init__(self, device, activation='sigmoid'):
        self.device = device
        self.activation = activation

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()

    def run_one_predict(self, model, images):
        predictions = model(images)
        if self.activation == 'sigmoid':
            predictions = F.sigmoid(predictions)
        elif self.activation == 'softmax':
            predictions = predictions.exp()
        return predictions

    def predict(self, model, loader):
        model = model.to(self.device).eval()

        with torch.no_grad():
            for data in loader:
                images = data.to(self.device)
                predictions = model(images)
                for prediction in predictions:
                    prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                    yield prediction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(args.config)
    paths = get_config(args.paths)
    params = config['train_params']
    model_name = config['train_params']['model']
    model = pydoc.locate(model_name)(**params['model_params'])
    model.load_state_dict(torch.load(params['weights'])['state_dict'])
    paths = paths['data']

    dataset = TestDataset(
        path=Path(paths['path']),
        image_csv=pd.read_csv(os.path.join(paths['path'], paths['test_images'])),
        transform=test_transform(**config['data_params']['augmentation_params']))

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=16,
        pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = PytorchInference(device)
    torch.set_num_threads(20)
    test_csv = pd.read_csv(os.path.join(paths['path'], paths['test_images']))
    print(os.path.join(paths['path'], paths['test_images']))
    #test_csv = test_csv[test_csv["fold"]==0]
    test_csv['predicted_EncodedPixels'] = None
    i = 0
    THRESHOLD = [0.75]
    COLUMN_WITH_PREDICTIONS = "Prediction_Yaroslav_resnet50_0.75_2"

    for pred in loader:
        print(pred)
        print(pred.shape)
        #test_csv.loc[i, COLUMN_WITH_PREDICTIONS] = pred
        test_csv.loc[i, COLUMN_WITH_PREDICTIONS] = mask_to_rle(pred.item)
        i += 1
    test_csv.to_csv(os.path.join(paths['path'], paths['test_images']), index=False)

    # for pred in tqdm(inferencer.predict(model, loader), total=len(dataset)):
    #     for threshold in THRESHOLD:
    #         pred_picture = (cv2.resize(pred, dsize=(1024, 1024), interpolation=cv2.INTER_LANCZOS4) > threshold).astype(
    #             int)
    #         test_csv.loc[i, COLUMN_WITH_PREDICTIONS] = mask_to_rle(pred_picture)  # for segmentation
    #         # test_csv.loc[i, COLUMN_WITH_PREDICTIONS] = -1 if pred[1] < threshold else 0 # for classification
    #         # if (np.sum(pred_picture) > 0):
    #         #     plt.imshow(pydicom.dcmread(os.path.join(paths['path'], paths['test_images'], test_csv.loc[i, 'ImageId'] + '.dcm')).pixel_array)
    #         #     plt.imshow(pred_picture, alpha=0.3)
    #         #     plt.savefig('/home/nkotelevskii/PyCharm/SIIM_project/src/inference_pics/{}_{}.png'.format(i, theshold))
    #     i += 1
    # test_csv.to_csv(os.path.join(paths['path'], paths['test_images']), index=False)


if __name__== '__main__':
    main()
