import copy
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
%matplotlib inline

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)

file_path = "../../data/ibespalov/SIIM_ACR/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.300.1517875162.258080/1.2.276.0.7230010.3.1.3.8323329.300.1517875162.258079/1.2.276.0.7230010.3.1.4.8323329.300.1517875162.258081.dcm"
csv_path = "../../data/ibespalov/SIIM_ACR/train-rle.csv"

def init_img_mask(file_path=file_path, csv_path=csv_path):
    dataset = pydicom.dcmread(file_path)
    init_img = dataset.pixel_array
    st = file_path.split(sep = "/")[-1][:-4]
    mask = pd.read_csv(csv_path)
    RLE_mask = mask.loc[mask['ImageId'] == st][" EncodedPixels"].iloc[0]
    rle_mask = rle2mask(RLE_mask[1:], 1024, 1024)
    init_with_mask = copy.deepcopy(dataset.pixel_array)
    init_with_mask[np.where(rle_mask)] = 1
    return init_img, rle_mask, init_with_mask
    
def plots(*args): #init_img, rle_mask, init_with_mask):
    init_img=args[0]
    rle_mask=args[1]
    with_mask=args[2]
    fig, ax = plt.subplots(1, 3, figsize=(15,12))
    ax[0].imshow(init_img, cmap=plt.cm.bone)
    ax[1].imshow(rle_mask, cmap=plt.cm.bone)
    ax[2].imshow(with_mask, cmap=plt.cm.bone)
    plt.show()
    
    
