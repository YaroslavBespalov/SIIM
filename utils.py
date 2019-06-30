import copy
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# %matplotlib inline

def mask_to_rle(img, width, height):
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

file_path = "../../data/ibespalov/SIIM_ACR/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.1851.1517875169.919022/1.2.276.0.7230010.3.1.3.8323329.1851.1517875169.919021/1.2.276.0.7230010.3.1.4.8323329.1851.1517875169.919023.dcm"
#file_path = "../../data/ibespalov/SIIM_ACR/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.302.1517875162.286329/1.2.276.0.7230010.3.1.3.8323329.302.1517875162.286328/1.2.276.0.7230010.3.1.4.8323329.302.1517875162.286330.dcm"

csv_path = "../../data/ibespalov/SIIM_ACR/train-rle.csv"



def init_img_mask(file_path=file_path, csv_path=csv_path, mask_id=0):
    dataset = pydicom.dcmread(file_path)
    init_img = dataset.pixel_array
    st = file_path.split(sep = "/")[-1][:-4]
    mask = pd.read_csv(csv_path)
    RLE_mask = mask.loc[mask['ImageId'] == st][" EncodedPixels"].values[mask_id]
    if RLE_mask.strip() != str(-1):
        rle_mask = rle2mask(RLE_mask[1:], 1024, 1024).T
    else:
        rle_mask = np.zeros((1024, 1024))
    init_with_mask = copy.deepcopy(dataset.pixel_array)
    init_with_mask[np.where(rle_mask)] = 1
    return init_img, rle_mask, init_with_mask

def plots(arg): #init_img, rle_mask, init_with_mask):
    init_img, rle_mask, with_mask = arg
    fig, ax = plt.subplots(1, 3, figsize=(20,15))
    ax[0].imshow(init_img, cmap=plt.cm.bone)
    ax[1].imshow(rle_mask, cmap=plt.cm.bone)
    ax[2].imshow(with_mask, cmap=plt.cm.bone)
    plt.show()
