import glob
import shutil
import os

PATH = '../../data/ibespalov/SIIM_ACR'

copy_path = '{}/train_samples'.format(PATH)
train_path = '{}/dicom-images-train/*/*/*.dcm'.format(PATH)

for file_name in glob.glob(train_path):
    tmp = os.path.join(copy_path, file_name.split(sep="/")[-1])
    shutil.copy2(file_name, tmp)

copy_path = '{}/test_samples'.format(PATH)
test_path = '{}/dicom-images-test/*/*/*.dcm'.format(PATH)

for file_name in glob.glob(test_path):
    tmp = os.path.join(copy_path, file_name.split(sep="/")[-1])
    shutil.copy2(file_name, tmp)
