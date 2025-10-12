import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import csv
import os

# Windows compatible paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'miniimagenet')
SPLIT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'miniimagenet', 'splits') + os.sep
assert os.path.exists(DATA_PATH), f'DATA_PATH does not exist: {DATA_PATH}'
os.makedirs(SPLIT_PATH, exist_ok=True)
split_list = ['meta_train', 'meta_val', 'meta_test']

if __name__=='__main__':
    for split in split_list:
        class_file_list=[]
        path_list = []
        label_list = []
        split_path=join(DATA_PATH,split)
        class_list=[f for f in listdir(split_path) if isdir(join(split_path, f))]
        for class_name in class_list:
            class_path=join(split_path,class_name)
            class_file_list.append([ join(class_path, cf) for cf in listdir(class_path) if (isfile(join(class_path,cf)) and cf[0] != '.')])
        for i, file_list in enumerate(class_file_list):
            path_list=path_list+file_list
            context = os.path.basename(os.path.dirname(file_list[0]))
            label_list = label_list + np.repeat(context, len(file_list)).tolist()


        fo = open(SPLIT_PATH + split + ".csv", "w",newline='')
        writer = csv.writer(fo)
        writer.writerow(['filename','label'])
        temp=np.array(list(zip(path_list,label_list)))
        writer.writerows(temp)
        fo.close()
        print("%s -OK" %split)
