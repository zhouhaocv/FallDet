from flask import Flask, request, jsonify
import json
import base64
import numpy as np
import cv2
import os
import time
from matplotlib.path import Path
import requests
from PIL import Image
import io
import copy
import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer)

app = Flask(__name__)

weights = './weights/yolov5s.pt'
imgsize = 960
confthres = 0.4
iouthres = 0.5
frame_number = 8
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

# Initialize
device = select_device('')
half = device.type != 'cpu'
model = attempt_load(weights, map_location=device)
names = model.module.names if hasattr(model, 'module') else model.names
if half:
    model.half()  # to FP16
imgsz = check_img_size(imgsize, s=model.stride.max())
# image data list
imagedatalist = []
# result list
resultlist = []
# detected index
detected = []
label = ['walking']


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def imgdecoder(img):
    img_b64decode = base64.b64decode(img)  # base64解码
    image = io.BytesIO(img_b64decode)
    face_image = Image.open(image)
    image_data = np.array(face_image)
    return image_data


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


@app.route('/FallFilter', methods=['POST'])
def FallFilter():
    """Detecting Procedure

    This is a simple implementation of pedestrian detection with HTTP protocol.
    Images will be saved in folder 'row-img' and 'pre-img' with the name '%time.jpg'.

    Args:
        data: {
            image_data: base64,
            image_id: int,
            ignore_area: [[{x:, y: }, {}, ..., {}], [], ..., []]
        }

    Returns:
        data: {
            code: int
            ret: int
            area: [{x:, y:, w:, h:}, {}, {}]
        }
        code: it will be assigned with 0, 1, -1, in which 0: success;
                                                          1: data receiving failed;
                                                          -1: detecting procedure failed.
        ret: confidence score.
        area: detecting results.
    """

    # receive data
    a = time.time()
    if not request.get_data():
        data = {
            'code': 2  # no input data
        }
        return jsonify(data)
    data = json.loads(request.get_data())
    if data['sf']:
        imagedatalist.clear()
        detected.clear()
        resultlist.clear()
        label.clear()
        label.append('walking')
    image_data = data['img']
    image_id = data['id']
    if data['roi'] == "":
        roi_region = [[(0, 0), (1280, 0), \
                       (1280, 720), (0, 720)]]
    else:
        roi = data['roi'].split(' ')
        roi_region = [[(int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), \
                       (int(roi[4]), int(roi[5])), (int(roi[6]), int(roi[7]))]]

    roi_region = [[(0, 0), (1280, 0), (1280, 720), (0, 720)]]
    print(roi_region)

    # base64 img to cv2
    imgbytes = base64.b64decode(image_data)
    img_array = np.fromstring(imgbytes, np.uint8)
    img0 = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    height, width, _ = img0.shape[0], img0.shape[1], img0.shape[2]
    img = letterbox(img0, new_shape=imgsize)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # detection process
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, confthres, iouthres, classes=None, agnostic=False)

    result = []

    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                if cls == 0:
                    centerDetector = Path(roi_region[0])
                    isContain = centerDetector.contains_points([(xywh[0], xywh[1])])
                    if isContain[0]:
                        result.append(xyxy)

    if len(result):
        if len(detected) < frame_number:
            detected.append(1)
        else:
            detected.pop(0)
            detected.append(1)
    else:
        if len(detected) < frame_number:
            detected.append(0)
        else:
            detected.pop(0)
            detected.append(0)
    # detected frame number > 0.5 * frame_number
    if len(imagedatalist) < frame_number:
        imagedatalist.append(image_data)
        resultlist.append(result)
        data = {
            'sf': False,
            'person': False,
            'id': data['id'],
            'code': 0,
            'class': label[0]
        }
        # eturn jsonify(data)
    else:
        imagedatalist.pop(0)
        imagedatalist.append(image_data)
        resultlist.pop(0)
        resultlist.append(result)
    # print(detected)

    if sum(detected) >= int(0.5 * frame_number) and len(imagedatalist) == frame_number:
        # send to action server
        send_image = copy.deepcopy(imagedatalist)
        imagedatalist.clear()
        detected.clear()
        print('imagedatalist', len(imagedatalist), len(send_image))
        '''
        for i in range(len(imagedatalist)-1):
            send_image += imagedatalist[i] + '****'
        send_image += imagedatalist[-1]
        '''
        send = {
            'image': send_image,
            'camera_id': data['camera_id'],
            'box': resultlist,
            'frame_number': frame_number,
        }

        r = requests.post('http://192.168.1.103:9988/AlertAction', data=json.dumps(send, cls=MyEncoder))
        resultlist.clear()
        r = json.loads(r.text)

        if r['code'] == 0:
            label.pop(0)
            label.append(r['class'])
            for re in result:
                x, y, w, h = int(re[0] - re[2] / 2), int(re[1] - re[3] / 2), int(re[2]), int(re[3])
                cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 0, 255), 2)
            base64_str = cv2.imencode('.jpg', img0)[1].tostring()
            base64_str = base64.b64encode(base64_str)
            base64_str = base64_str.decode('utf-8')

            data = {
                'sf': False,
                'person': True,
                'id': data['id'],
                'code': 0,
                'class': label[0],
                'img': base64_str
            }

            return jsonify(data)
        else:
            data = {
                'sf': False,
                'person': False,
                'code': r['code'],
                'id': data['id'],
                'class': label[0]
            }
            return jsonify(data)
    else:
        data = {
            'sf': False,
            'person': False,
            'id': data['id'],
            'code': 0,
            'class': label[0]
        }
        return jsonify(data)


if __name__ == '__main__':
    app.run('0.0.0.0', port=7880)
