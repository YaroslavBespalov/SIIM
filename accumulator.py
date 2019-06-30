import glob
import shutil
import os

copy_path = '../../data/ibespalov/SIIM_ACR/train_samples'
train_path = '../../data/ibespalov/SIIM_ACR/dicom-images-train/*/*/*.dcm'

for file_name in glob.glob(train_path):
    tmp = os.path.join(copy_path, file_name.split(sep="/")[-1])
    shutil.copy2(file_name, tmp)

copy_path = '../../data/ibespalov/SIIM_ACR/test_samples'
test_path = '../../data/ibespalov/SIIM_ACR/dicom-images-test/*/*/*.dcm'

for file_name in glob.glob(test_path):
    tmp = os.path.join(copy_path, file_name.split(sep="/")[-1])
    shutil.copy2(file_name, tmp)
