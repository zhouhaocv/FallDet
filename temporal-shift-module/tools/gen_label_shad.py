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
dataset_name = 'SHAD3-crop'
label_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-labels'
dataset_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames' # 'jester-v1'
if not os.path.exists(label_path):
    os.makedirs(label_path)
if __name__ == '__main__':
    
    categories = ['fall','stand','squat','lying','sit']
    idx_categories = {'fall':0,'stand':1,'squat':2,'lying':3,'sit':4}
    train_output = 'train_videofolder.txt'
    crop_name = ['crop1','crop2','crop3','crop4']
    val_output = 'val_videofolder.txt'
    output1 = []
    output2 = []
    for ii in range(len(categories)):
        img_dir1 = os.path.join(dataset_path,categories[ii])
        folders = os.listdir(img_dir1)
        random.shuffle(folders)

        for i in range((4*len(folders)//5)):
            curFolder = folders[i]
            curIDX = idx_categories[categories[ii]]
            crop_dir = os.path.join(dataset_path,categories[ii], curFolder)
            crop_list = os.listdir(crop_dir)
            crop_list = crop_list[1:]
            for k in range(len(crop_list)):
                img_dir2 = os.listdir(crop_dir+'/'+crop_list[k])

                for jj in range(len(img_dir2)):
                    dir_files = os.listdir(os.path.join(crop_dir+'/'+crop_list[k],img_dir2[jj]))

                    output1.append('%s %d %d'%(os.path.join('videos_trimmed/'+ dataset_name + '-frames',categories[ii], curFolder+'/'+crop_list[k],img_dir2[jj]), len(dir_files), curIDX))
                    print('%d/%d'%(i, len(folders)))

        for i in range((4*len(folders)//5),len(folders)):
        # for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[categories[ii]]
            crop_dir = os.path.join(dataset_path,categories[ii], curFolder)
            crop_list = os.listdir(crop_dir)
            crop_list = crop_list[1:]
            for k in range(len(crop_list)):
                img_dir2 = os.listdir(crop_dir+'/'+crop_list[k])

                for jj in range(len(img_dir2)):
                    dir_files = os.listdir(os.path.join(crop_dir+'/'+crop_list[k],img_dir2[jj]))

                    output2.append('%s %d %d'%(os.path.join('videos_trimmed/'+ dataset_name + '-frames',categories[ii], curFolder+'/'+crop_list[k],img_dir2[jj]), len(dir_files), curIDX))
                    print('%d/%d'%(i, len(folders)))

    with open(os.path.join(label_path, train_output),'w') as f:
        f.write('\n'.join(output1))
    with open(os.path.join(label_path, val_output),'w') as f:
        f.write('\n'.join(output2))