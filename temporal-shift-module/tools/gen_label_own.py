# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
import random

label_path = '/raid/file2/code2/tsm/data/FallDet/labels'
dataset_path = '/raid/file2/code2/tsm/data/FallDet' # 'jester-v1'
if not os.path.exists(label_path):
    os.makedirs(label_path)
if __name__ == '__main__':

    categories = ['fall','stand','squat','lying','sit']
    idx_categories = {'fall':0,'stand':1,'squat':2,'lying':3,'sit':4}
    train_output = 'train_videofolder.txt'
    val_output = 'val_videofolder.txt'
    output1 = []
    output2 = []
    for ii in range(len(categories)):
        img_dir1 = os.path.join(dataset_path,categories[ii])
        folders = os.listdir(img_dir1)
        random.shuffle(folders)
        if categories[ii] == 'stand':
            train_num = 2*len(folders)//4
            val_num = 2*len(folders)//4
        else:
            train_num = 3*len(folders)//4
            val_num = 3*len(folders)//4

        for i in range(train_num):
            curFolder = folders[i]
            curIDX = idx_categories[categories[ii]]
            img_dir = os.path.join(dataset_path, categories[ii], curFolder)

            dir_files = os.listdir(img_dir)
            dir_files = [x for x in dir_files if x.endswith('.jpg')]

            output1.append('%s %d %d'%(os.path.join(categories[ii], curFolder), len(dir_files), curIDX))
            print('%d/%d'%(i, len(folders)))

        for i in range(val_num,len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[categories[ii]]
            img_dir = os.path.join(dataset_path,categories[ii], curFolder)

            dir_files = os.listdir(img_dir)   
            dir_files = [x for x in dir_files if x.endswith('.jpg')]


            output2.append('%s %d %d'%(os.path.join(categories[ii], curFolder), len(dir_files), curIDX))
            print('%d/%d'%(i, len(folders)))

    with open(os.path.join(label_path, train_output),'w') as f:
        f.write('\n'.join(output1))
    with open(os.path.join(label_path, val_output),'w') as f:
        f.write('\n'.join(output2))