# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

# Notice that this file has been modified to support ensemble testing
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont

from YOLOv4.models import Yolov4
from YOLOv4.tool.utils import load_class_names
from YOLOv4.tool.torch_utils import do_detect
import cv2

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)

parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

parser.add_argument('--vis', default=False, action="store_true", help='visualization')
parser.add_argument('--yolo_model', type=str, default=None, help='yolo model for human detection')
parser.add_argument('--arch', type=str, default="BNInception")

args = parser.parse_args()

#classnames = ["panpa", "paotou", "qiaoqiang", "zhengchang", "zhiliu"]
classnames = ["可疑物品", "攀爬", "抛投异物", "敲墙", "正常行为", "滞留"]
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]

error_count = 0
num_classes = 80
width = 608
height = 608
scale_size = 720

def yolo_detect(m, input):
    if num_classes == 20:
        namesfile = 'YOLOv4/data/voc.names'
    elif num_classes == 80:
        namesfile = 'YOLOv4/data/coco.names'
    else:
        namesfile = 'YOLOv4/data/x.names'
    class_names = load_class_names(namesfile)

    # take the first one(person)
    class_names = class_names[0]

    boxes_list = []

    for i in range(len(input)):
        for j in range(0, len(input[i]), 3):
            img = input[i, j:j+3].numpy()
            img = img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            img = img * 255
            img = img.astype(np.uint8)
            sized = cv2.resize(img, (width, height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(m, sized, 0.4, 0.6)
            selected_boxes = []
            for box in boxes[0]:
                if box[6] < len(class_names):
                    selected_boxes.append(box)
            boxes_list.append(selected_boxes)
            # print('yolo-time: %f seconds'  % (finish - start))

    return boxes_list

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


weights_list = args.weights.split(',')
test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
    coeff_list = [1] * len(weights_list)
else:
    coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
    test_file_list = args.test_list.split(',')
else:
    test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
    if 'RGB' in this_weights:
        modality = 'RGB'
    else:
        modality = 'Flow'
    this_arch = this_weights.split('TSM_')[1].split('_')[2]
    modality_list.append(modality)
    num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                            modality)
    print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
    net = TSN(num_class, this_test_segments if is_shift else 1, modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in this_weights
              )

    if 'tpool' in this_weights:
        from ops.temporal_shift import make_temporal_pool
        make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

    checkpoint = torch.load(this_weights)
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

    input_size = net.scale_size if args.full_res else net.input_size
    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            #GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size, flip=False)
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(input_size, net.scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
                       new_length=1 if modality == "RGB" else 5,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           #cropping,
                           Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                           #GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
    )

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    data_gen = enumerate(data_loader)

    if total_num is None:
        total_num = len(data_loader.dataset)
    else:
        assert total_num == len(data_loader.dataset)

    data_iter_list.append(data_gen)
    net_list.append(net)

output = []
transform_noBox = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            # !!! no cropping
            # GroupCenterCrop(crop_size),
            Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
            GroupNormalize(input_mean, input_std)
        ])

def eval_video(video_data, net, this_test_segments, modality, yolo_model):
    global error_count

    net.eval()
    with torch.no_grad():
        i, data, label = video_data
        data_org = data

        boxes_list = yolo_detect(yolo_model, data)
        content_flag = False
        for box in boxes_list:
            if box:
                content_flag = True
                break

        if content_flag:
            transform = torchvision.transforms.Compose([
                # GroupMultiPersonCrop(720, boxes_list),
                # !!! up sample to scale size
                GroupMultiPersonCrop(scale_size, boxes_list),
                # MultiPersonCrop(224,[0,0,100,100]),
                Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                GroupNormalize(input_mean, input_std),
            ])
            data = transform(data)
        else:
            img_group = []
            for i in range(0, len(data[0]), 3):
                img = data[0, i:i + 3]
                img = torchvision.transforms.ToPILImage()(img)
                img_group.append(img)
            data = transform_noBox(img_group)

        batch_size = label.numel()
        num_crop = args.test_crops
        if args.dense_sample:
            num_crop *= 10  # 10 clips for testing when using dense sample

        if args.twice_sample:
            num_crop *= 2

        if modality == 'RGB':
            length = 3
        elif modality == 'Flow':
            length = 10
        elif modality == 'RGBDiff':
            length = 18
        else:
            raise ValueError("Unknown modality "+ modality)

        data_in = data.view(-1, length, data.size(1), data.size(2))

        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            # take the softmax to normalize the output to probability
            rst = F.softmax(rst, dim=1)

        rst = rst.data.cpu().numpy().copy()

        if net.module.is_shift:
            rst = rst.reshape(batch_size, num_class)
        else:
            rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

        # visualization, blue for ground truth, red for prediction
        if args.vis:
            for i, video_data in enumerate(data_in):
                for j, img_data in enumerate(video_data):
                    img_data = GroupDeNormalize(input_mean, input_std)(img_data)
                    img_data = data_org[i,j*3:(j+1)*3]
                    img = img_data.numpy()
                    img = img.swapaxes(0, 1)
                    img = img.swapaxes(1, 2)
                    img = img * 255

                    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
                        img = Image.fromarray(np.uint8(img))
                        # 创建一个可以在给定图像上绘图的对象
                    draw = ImageDraw.Draw(img)
                    # 字体的格式
                    fontStyle = ImageFont.truetype(
                        "font/simsun.ttc", 30, encoding="utf-8")
                    # 绘制文本
                    draw.text((30, 60), "预测行为：" + classnames[np.argmax(rst[i])], (255, 0, 0), font=fontStyle)
                    draw.text((30, 10), "标注行为：" + classnames[label], (0, 0, 255), font=fontStyle)
                    # 转换回OpenCV格式
                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

                    print('prediction: ' + classnames[np.argmax(rst[i])])
                    print('ground truth: ' + classnames[label])
                    # if classnames[np.argmax(rst[i])] != classnames[label]:
                    #     cv2.imwrite('TSM_clipped_seg8_error/{}.jpg'.format(error_count), img)
                    #     cv2.imshow('imshow', img)
                    #     cv2.waitKey(40)
                    #     error_count += 1
                    cv2.imwrite('TSM_clipped_seg8_yolo_6class/{}.jpg'.format(error_count), img)
                    #cv2.imshow('imshow', img)
                    #cv2.waitKey(40)
                    error_count += 1
        return i, rst, label

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()

confusionMatrix = np.zeros((len(classnames), len(classnames)), dtype=int)

# Yolov4 model configuration
yolo_model = Yolov4(yolov4conv137weight=None, n_classes=num_classes, inference=True)

pretrained_dict = torch.load(args.yolo_model, map_location=torch.device('cuda'))
yolo_model.load_state_dict(pretrained_dict)
yolo_model.cuda()

for i, data_label_pairs in enumerate(zip(*data_iter_list)):
    with torch.no_grad():
        if i >= max_num:
            break
        this_rst_list = []
        this_label = None
        for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
            # print('The shape of the input data: ')
            # print(data.shape)
            rst = eval_video((i, data, label), net, n_seg, modality, yolo_model)
            this_rst_list.append(rst[1])
            this_label = label
        assert len(this_rst_list) == len(coeff_list)
        for i_coeff in range(len(this_rst_list)):
            this_rst_list[i_coeff] *= coeff_list[i_coeff]
        ensembled_predict = sum(this_rst_list) / len(this_rst_list)

        for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
            output.append([p[None, ...], g])

        confusionMatrix[int(this_label.cpu().numpy()), int(np.where(ensembled_predict == np.max(ensembled_predict))[1])] += 1

        cnt_time = time.time() - proc_start_time
        prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
        top1.update(prec1.item(), this_label.numel())
        top5.update(prec5.item(), this_label.numel())
        if i % 20 == 0:
            print('video {} done, total {}/{}, average {:.3f} sec/video, '
                  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                              float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))

video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]


if args.csv_file is not None:
    print('=> Writing result to csv file: {}'.format(args.csv_file))
    with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
        categories = f.readlines()
    categories = [f.strip() for f in categories]
    with open(test_file_list[0]) as f:
        vid_names = f.readlines()
    vid_names = [n.split(' ')[0] for n in vid_names]
    assert len(vid_names) == len(video_pred)
    if args.dataset != 'somethingv2':  # only output top1
        with open(args.csv_file, 'w') as f:
            for n, pred in zip(vid_names, video_pred):
                f.write('{};{}\n'.format(n, categories[pred]))
    else:
        with open(args.csv_file, 'w') as f:
            for n, pred5 in zip(vid_names, video_pred_top5):
                fill = [n]
                for p in list(pred5):
                    fill.append(p)
                f.write('{};{};{};{};{};{}\n'.format(*fill))


cf = confusion_matrix(video_labels, video_pred).astype(float)

np.save('cm.npy', cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt
print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

print(confusionMatrix)


