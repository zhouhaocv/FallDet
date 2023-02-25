# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading

dataset_name = 'multiple_fall'
VIDEO_ROOT1 = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name        # Downloaded webm videos
FRAME_ROOT1 = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames'  # Directory for extracted frames


def extract(video, tmpl='%04d.jpg'):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    cmd = 'ffmpeg -i \"{}/{}\" -threads 1 -vf scale=-1:720 -q:v 0 -r 25 \"{}/{}/%04d.jpg\"'.format(VIDEO_ROOT, video,
                                                                                             FRAME_ROOT, video[:-4])
    os.system(cmd)


def target(video_list):
    for video in video_list:
        os.makedirs(os.path.join(FRAME_ROOT, video[:-4]))
        extract(video)


if __name__ == '__main__':
    categories = ['fall','stand','squat','lying','sit']
    for ii, category_file in enumerate(categories):
        VIDEO_ROOT = os.path.join(VIDEO_ROOT1, category_file)
        FRAME_ROOT = os.path.join(FRAME_ROOT1, category_file)
        if not os.path.exists(VIDEO_ROOT):
            raise ValueError('Please download videos and set VIDEO_ROOT variable.')
        if not os.path.exists(FRAME_ROOT):
            os.makedirs(FRAME_ROOT)

        video_list = os.listdir(VIDEO_ROOT)
        target(video_list)