import random

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os

# Windows compatible paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'euroSAT', 'EuroSAT')
SPLIT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'euroSAT', 'splits') + os.sep
assert os.path.exists(DATA_PATH), f'DATA_PATH does not exist: {DATA_PATH}'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

SPLIT={
'meta_train': ['AnnualCrop' , 'Forest' , 'HerbaceousVegetation' , 'Highway' , 'Industrial' , 'Pasture' , 'PermanentCrop',  'Residential' , 'River' , 'SeaLake'],
    'meta_val': [],
    'meta_test': [],
}

folder_list = [f for f in listdir(DATA_PATH) if isdir(join(DATA_PATH, f))]
folder_list.sort()
label_dict = dict(zip(range(0,len(folder_list)),folder_list))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(DATA_PATH, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.' )])
    random.shuffle(classfile_list_all[i])


for split in split_list:
    num=0
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if label_dict[i] in SPLIT[split]:
            file_list = file_list + classfile_list
            label_list = label_list + np.repeat(label_dict[i], len(classfile_list)).tolist()
            num = num + 1
    print('split_num:',num)
    fo = open(SPLIT_PATH + split + ".csv", "w",newline='')
    writer = csv.writer(fo)
    writer.writerow(['filename','label'])
    temp=np.array(list(zip(file_list,label_list)))
    writer.writerows(temp)
    fo.close()
    print("%s -OK" %split)