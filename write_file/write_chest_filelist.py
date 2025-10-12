import numpy as np
import csv
import os
import pandas as pd

DATA_PATH=''
SPLIT_PATH = ''
assert len(DATA_PATH)!=0,'You should input the data_path!'
assert len(SPLIT_PATH)!=0,'You should input the savedir!'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']
used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
data_info = pd.read_csv('Path_to_Data_Entry_2017.csv', skiprows=[0], header=None)
# First column contains the image paths
image_name_all = [os.path.join(DATA_PATH,i) for i in np.asarray(data_info.iloc[:, 0])]
# First column contains the image paths
labels_all = np.asarray(data_info.iloc[:, 1])

image_name  = []
labels = []


for name, label in zip(image_name_all,labels_all):
    label = label.split("|")
    if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in used_labels:
        labels.append(labels_maps[label[0]])
        image_name.append(name)


path_label=list(zip(image_name,labels))
path_label.sort(key=lambda a: a[1])
print(np.array(path_label))
fo = open(SPLIT_PATH + 'meta_test' + ".csv", "w",newline='')
writer = csv.writer(fo)
writer.writerow(['filename','label'])
writer.writerows(np.array(path_label))
fo.close()
print("%s -OK" %'meta_test')

