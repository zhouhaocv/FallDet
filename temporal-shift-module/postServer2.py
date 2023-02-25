from flask import Flask, request, jsonify
import json
import base64
import numpy as np
import cv2
import os
import time
from ops.models import TSN
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from ops.transforms import *
import threading

app = Flask(__name__)

weight_file = 'SOTA_model/TSM_falldet2classes_RGB_resnet50_shift8_blockres_avg_segment8_e50_v2/ckpt.best.pth.tar'
num_class = 2
test_segments = 8
img_feature_dim = 256
test_crops = 1
full_res = False
pretrain = 'imagenet'
dense_sample = False
twice_sample = False
softmax = False
classnames = ['fall', 'normal']
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
gpus = ['7']
workers = 1
scale_size = 720
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]


def eval_video(video_data, net, this_test_segments, modality):
    with torch.no_grad():
        data = video_data
        batch_size = 1
        num_crop = test_crops
        if dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality " + modality)

        print("data_in shape:")
        print(data.shape)
        data_in = data.view(-1, length, data.size(1), data.size(2))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if True:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        return rst


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weight_file)
if 'RGB' in weight_file:
    modality = 'RGB'
else:
    modality = 'Flow'
this_arch = weight_file.split('TSM_')[1].split('_')[2]
print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
net = TSN(num_class, test_segments if is_shift else 1, modality,
          base_model=this_arch,
          consensus_type='avg',
          img_feature_dim=img_feature_dim,
          pretrain=pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in weight_file,
          )
if 'tpool' in weight_file:
    from ops.temporal_shift import make_temporal_pool

    make_temporal_pool(net.base_model, test_segments)  # since DataParallel

checkpoint = torch.load(weight_file)
checkpoint = checkpoint['state_dict']

# base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias',
                }
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)

input_size = net.scale_size if full_res else net.input_size
if test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        # GroupCenterCrop(input_size),
    ])
elif test_crops == 3:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupFullResSample(input_size, net.scale_size, flip=False)
    ])
elif test_crops == 5:  # do not flip, so only 5 crops
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size, flip=False)
    ])
elif test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(test_crops))

if gpus is not None:
    devices = [gpus[i] for i in range(workers)]
else:
    devices = list(range(workers))

net = torch.nn.DataParallel(net.cuda())
net.eval()


@app.route('/AlertAction', methods=['POST'])
def AlertAction():
    """Recognition Procedure

    This is a simple implementation of action recognition with HTTP protocol.
    Receive 8 images that separated by ****, and use TSM to recognize the action class.

    Args:
        data: {
            image: base64,
            camera_id: int,
            frame_number: int
        }
        box: [[{x:, y: }, {}, ..., {}], [], ..., []

    Returns:
        data: {
            code: int; -1: frame num error; 0: normal; 1: model error, 2: transmission error,
            class: string,
            camera_id: int
        }
        code: it will be assigned with 0, 1, -1, 2, in which 0: success;
                                                          1: model recognition failed;
                                                          -1: frame number mismatch;
                                                          2: transmission error.
        ret: confidence score.
        area: detecting results.
    """
    proc_start_time = time.time()
    if not request.get_data():
        print("Error: failed to receive image data")
        return_data = {
            'code': 2
        }
        return jsonify(return_data)
    data = json.loads(request.get_data())
    image_list = data['image']
    # print("Box:")
    # print(data['box'])
    if image_list is None:
        print("Error: receive no image data")
        return_data = {
            'code': 2
        }
        return jsonify(return_data)

    frame_num = len(image_list)
    if not frame_num == data['frame_number']:
        print("Error: frame num receive: " + str(frame_num))
        return_data = {
            'code': -1
        }
        return jsonify(return_data)
    preprocess_time = time.time() - proc_start_time
    print('preprocess time: {:.3f}'.format(float(preprocess_time)))

    box_list = data['box']

    proc_start_time = time.time()
    image_list_PIL = []
    for i in range(len(image_list)):
        image = base64.b64decode(image_list[i])
        img_array = np.fromstring(image, np.uint8)

        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image_list_PIL.append(img)

    transform = torchvision.transforms.Compose([
        GroupMultiPersonCrop(scale_size, box_list),
        # cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(input_mean, input_std)
    ])
    image_all = transform(image_list_PIL)

    decode_crop_time = time.time() - proc_start_time
    print('decode and crop time: {:.3f}'.format(float(decode_crop_time)))

    net.eval()
    try:
        proc_start_time = time.time()
        results = eval_video(image_all, net, test_segments, modality)
        cnt_time = time.time() - proc_start_time
        print('{:.3f} sec/video'.format(float(cnt_time)))
    except Exception as e:
        print(e)
        print("Error: model running error")
        return_data = {
            'code': 1
        }
        return jsonify(return_data)

    results = results[0].tolist()
    classIndex = results.index(max(results))
    if len(results):
        print("The return result: " + classnames[classIndex])
        print('probability:',results[classIndex])
        return_data = {
            'code': 0,
            'class': classnames[classIndex],
            'camera_id': data['camera_id']
        }
        return jsonify(return_data)
    else:
        print("Error: model running error")
        return_data = {
            'code': 1
        }
        return jsonify(return_data)


if __name__ == '__main__':
    # app.debug = True
    app.run('0.0.0.0', port=7988)
