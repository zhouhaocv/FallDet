# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import threading
from moviepy.editor import VideoFileClip
import shutil

NUM_THREADS = 100
dataset_name = 'URFD'
VIDEO_ROOT = '/raid/file2/code2/tsm/data/FallDet/videos_untrimmed/' + dataset_name        # Downloaded webm videos
TRIMMED_ROOT = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name  # Directory for extracted frames


def extract(video,duration):
    # os.system(f'ffmpeg -i {VIDEO_ROOT}/{video} -vf -threads 1 -vf scale=-1:256 -q:v 0 '
    #           f'{FRAME_ROOT}/{video[:-5]}/{tmpl}')
    #mpeg to h264
    # ffmpeg -i SHAD_001.mp4 -c:v libx264 -preset ultrafast -qp 0 -strict -2 SHAD_0001.mp4
    step = 2
    for start in range(0,int(duration),step):
        end = start + step
        cmd = 'ffmpeg -i \"{}/{}\" -ss {} -c copy -to {} -intra \"{}/{}/{}_{}_{:04d}.avi\"'.format(VIDEO_ROOT,video,start, end,
                                                                                                 TRIMMED_ROOT,video[:-4],dataset_name, video[:-4],start)
        print(cmd)
        os.system(cmd)


def target(video):
    # for video in video_list:
    if not os.path.exists(os.path.join(TRIMMED_ROOT, video[:-4])):
        os.makedirs(os.path.join(TRIMMED_ROOT, video[:-4]))
    clip = VideoFileClip(VIDEO_ROOT+'/'+video)
    duration = clip.duration
    extract(video,duration)


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(TRIMMED_ROOT):
        os.makedirs(TRIMMED_ROOT)

    video_list = os.listdir(VIDEO_ROOT)

    for ii in range(len(video_list)):
        target(video_list[ii])