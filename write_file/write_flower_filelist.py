import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import csv

# Windows compatible paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'vggflower', 'images')
SPLIT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'vggflower', 'splits') + os.sep
os.makedirs(SPLIT_PATH, exist_ok=True)

split_list = ['meta_train', 'meta_val', 'meta_test']

# Check if we have imagelabels.mat
label_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'vggflower', 'imagelabels.mat')
use_mat_file = os.path.exists(label_path)

if use_mat_file:
    import scipy.io
    labels_mat = scipy.io.loadmat(label_path)
    
    SPLITS = {
        'meta_train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
                  78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
                  63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
                  84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
                  33, 87, 1, 49, 20, 25, 58],
        'meta_val': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
        'meta_test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
        'all': list(range(1, 103)),
    }
    print(len(SPLITS['meta_train']))
    print(len(SPLITS['meta_val']))
    print(len(SPLITS['meta_test']))
    
    for mode in split_list:
        num = 0
        file_list = []
        label_list = []
        split = SPLITS[mode]
        for idx, label in enumerate(labels_mat['labels'][0], start=1):
            if label in split:
                file_list.append(os.path.join(DATA_PATH,'image_%05d.jpg'%(idx)))
                label_list.append(label)
                num=num+1
        print('split_num:', num)
        fo = open(SPLIT_PATH + mode + ".csv", "w", newline='')
        writer = csv.writer(fo)
        writer.writerow(['filename', 'label'])
        temp = np.array(list(zip(file_list, label_list)))
        writer.writerows(temp)
        fo.close()
        print("%s -OK" % mode)
else:
    # Fallback: scan all images and create labels based on filename patterns
    print(f"Using image file scanning (imagelabels.mat not found)")
    
    # Get all jpg files
    if os.path.exists(DATA_PATH):
        all_files = sorted([f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f)) and f.endswith('.jpg')])
        
        # Extract image numbers and assign labels
        # Assuming 80 images per class, 102 classes total (8189 images)
        images_per_class = 80
        
        file_list_all = []
        label_list_all = []
        
        for f in all_files:
            # Extract number from image_xxxxx.jpg
            num = int(f.split('_')[1].split('.')[0])
            # Calculate class label (1-102)
            label = ((num - 1) // images_per_class) + 1
            file_list_all.append(os.path.join(DATA_PATH, f))
            label_list_all.append(label)
        
        # Split based on class label
        SPLITS = {
            'meta_train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
                      78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
                      63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
                      84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
                      33, 87, 1, 49, 20, 25, 58],
            'meta_val': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
            'meta_test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
        }
        
        for mode in split_list:
            file_list = []
            label_list = []
            split = SPLITS[mode]
            
            for f, l in zip(file_list_all, label_list_all):
                if l in split:
                    file_list.append(f)
                    label_list.append(l)
            
            print(f'{mode} split_num: {len(file_list)}')
            fo = open(SPLIT_PATH + mode + ".csv", "w", newline='')
            writer = csv.writer(fo)
            writer.writerow(['filename', 'label'])
            temp = np.array(list(zip(file_list, label_list)))
            writer.writerows(temp)
            fo.close()
            print("%s -OK" % mode)
    else:
        print(f"ERROR: DATA_PATH does not exist: {DATA_PATH}")
