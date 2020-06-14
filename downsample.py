import numpy as np
import pandas as pd
import os
import subprocess
from tqdm import tqdm

# DICOM to JPEG
import pydicom as dicom
import cv2
import PIL

workDir = '/Users/yuta/kaggle_melanoma'

train = pd.read_csv(os.path.join(workDir, 'train.csv'))
train.drop(['diagnosis', 'benign_malignant'], axis=1, inplace=True)
print(train.columns.values)
print(train.head())

# number of samples with nan values
sum(train.isna().sum(axis=1) > 0)

# number of malignant samples with nan values
sum((train.isna().sum(axis=1) > 0) & (train.target == 1))

# total number of malignant samples
sum(train.target == 1)

# Drop samples with nan values.
train = train.loc[train.isna().sum(axis=1) == 0]

malignant_image_names = train['image_name'].loc[train['target'] == 1].values
num_malignant = len(malignant_image_names)

np.random.seed(0)
benign_image_names = np.random.choice(
    train['image_name'].loc[train['target'] == 0].values,
    size=num_malignant,
    replace=False
)
num_benign = len(benign_image_names)

retain_image_names = np.concatenate(
    (malignant_image_names, benign_image_names),
    axis=None
)

retain_image_labels = np.concatenate(
    (np.repeat(1, num_malignant), np.repeat(0, num_benign)),
    axis=None
)

labels = pd.DataFrame({'id': retain_image_names, 'label': retain_image_labels})
labels.to_csv('retained_labels.csv', index=False)

retain_file_names = [image + '.dcm' for image in retain_image_names.tolist()]

# Extract selected files from a Zip archive.
subprocess.check_call(
    "unzip -j train.zip {} -d train_dcm".format(' '.join(retain_file_names)),
    shell=True
)

# Convert them to JPEG.
if not os.path.exists('train_jpg'):
    subprocess.check_call("mkdir train_jpg", shell=True)

for file in tqdm(retain_file_names):
    path_input = os.path.join('train_dcm', file)
    ds = dicom.dcmread(path_input)
    pixel_array_numpy = ds.pixel_array
    path_output = path_input.replace('dcm', 'jpg')
    cv2.imwrite(path_output, pixel_array_numpy)

# Delete DICOM files.
subprocess.check_call("rm -r train_dcm", shell=True)
