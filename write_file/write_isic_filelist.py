import numpy as np
import csv
import os
import pandas as pd

# Windows compatible paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'isic')
SPLIT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'isic', 'splits') + os.sep
assert os.path.exists(DATA_PATH), f'DATA_PATH does not exist: {DATA_PATH}'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

# Use the ISIC 2019 ground truth file
gt_path = os.path.join(DATA_PATH, 'ISIC_2019_Training_GroundTruth.csv')
data_info = pd.read_csv(gt_path)
# First column contains the image paths
# The ground truth CSV has format: image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK
image_names = data_info.iloc[:, 0].tolist()

# Get the diagnosis columns (all except first column which is image name)
labels_array = np.asarray(data_info.iloc[:, 1:])
# Find which column has 1.0 for each row
label_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
labels = [label_names[i] for i in (labels_array==1.0).argmax(axis=1)]

# Create correct paths: data/isic/LABEL/IMAGE.jpg
image_name = []
for img_name, label in zip(image_names, labels):
    correct_path = os.path.join(DATA_PATH, label, img_name + '.jpg')
    image_name.append(correct_path)

for split in ['meta_train']:
    fo = open(SPLIT_PATH + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    path_label=list(zip(image_name,labels))
    path_label.sort(key=lambda a: a[1])
    print(np.array(path_label))
    writer.writerows(np.array(path_label))
    fo.close()
    print("%s -OK" %split)