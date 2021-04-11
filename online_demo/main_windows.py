"""
本代码的来龙去脉：
讨论如何把代码从jetson移植到windows：https://github.com/mit-han-lab/temporal-shift-module/issues/114
讨论如何结果带来的报错：https://github.com/mit-han-lab/temporal-shift-module/issues/148#issuecomment-794694025
本人也稍微精简优化了代码
"""

import numpy as np
import os
from typing import Tuple
import io
import time
import cv2
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import onnx
from mobilenet_v2_tsm import MobileNetV2

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


# 该函数实际并没有使用到
def transform(frame: np.ndarray):
    # input frame : 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 224, 224) 0 ~ 1.0
    # print(f'frame shape is {frame.shape}')
    return frame


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 123.675, 116.28, 103.53 ; 58.395, 57.12, 57.375
    ])
    return transform


catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # "Zooming Out With Two Fingers"  # 26
]


def process_output(idx_, idx_history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:  # default REFINE_OUTPUT = True
        return idx_, idx_history

    max_hist_len = 30  # 20, max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = idx_history[-1]
    print(f'1 history[-1] is {idx_history[-1]}')

    # 把"Doing other things" 等同于 "No gesture"
    if idx_ == 0:
        idx_ = 2

    # history smoothing
    if idx_ != idx_history[-1]:
        if not (idx_history[-1] == idx_history[-2]):  # and idx_history[-2] == idx_history[-3]):
            idx_ = idx_history[-1]

    idx_history.append(idx_)
    idx_history = idx_history[-max_hist_len:]  # 只保留最新的max_hist_len个元素
    print(f'history length in process_output is {len(idx_history)}')
    print(f'2 history[-1] is {idx_history[-1]}')
    return idx_history[-1], idx_history


def get_executor(use_gpu=True):
    torch_module = MobileNetV2(n_class=27)
    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):  # checkpoint not downloaded
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = 'https://file.lzhu.me/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    torch_module.load_state_dict(torch.load("mobilenetv2_jester_online.pth.tar"))
    torch_inputs = (#torch.rand(1, 3, 224, 224),
                    torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7]))

    return torch_module, torch_inputs


def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    # set a lower resolution for speed up
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    WINDOW_NAME = 'Video Gesture Recognition'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    transform = get_transform()
    model, buffer = get_executor()
    model.eval()
    idx = 0
    idx_history = [2]  # 初始化inx_history，idx = 2代表假设一开始时是No gesture
    history_logit = []
    i_frame = -1

    print("Ready!")
    with torch.no_grad():
        while True:
            i_frame += 1
            _, img = cap.read()
            if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
                t1 = time.time()

                img_tran = transform([Image.fromarray(img).convert('RGB')])
                # print(f'img_tran.size() is {img_tran.size()}')  # torch.Size([3, 224, 224])
                input_var = img_tran.view(1, 3, img_tran.size(1), img_tran.size(2))  # add one more dimension
                # model need be fed two inputs
                outputs = model(input_var, *buffer)
                feat, buffer = outputs[0], outputs[1:]
                # print(f'feat is {feat} and size is {feat.size()}')
                # feat is tensor([[1.2658, 0.2161, 2.6009, 0.2375, 0.4619, -0.6576, 0.2948, -0.9396,
                #                  -0.8498, 0.2217, 0.1223, -1.4733, -1.3406, 0.2248, 0.4350, 0.0632,
                #                  -1.3974, -1.4234, 0.2380, -0.0771, 0.2499, -0.2796, -0.0700, 0.3558,
                #                  0.4856, 0.4934, 0.5418]]) and size is torch.Size([1, 27])
                # print(f'buffer is {len(buffer)} and length is {len(buffer)}')  # 10
                feat = feat.detach()

                # 我看不出下面这个代码块计算出的idx_有什么用，它只求单次结果的最大值，误差大，所以先注释掉了
                # if SOFTMAX_THRES > 0:
                #     feat_np = feat.asnumpy().reshape(-1)
                #     feat_np -= feat_np.max()
                #     softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                #
                #     print(max(softmax))
                #     if max(softmax) > SOFTMAX_THRES:
                #         idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                #     else:
                #         idx_ = idx
                # else:
                #     idx_ = np.argmax(feat.cpu().numpy(), axis=1)[0]
                #     print(idx_)  # indicate the largest probability of gesture among 27 categories

                if HISTORY_LOGIT:  # default HISTORY_LOGIT = True
                    history_logit.append(feat.cpu().numpy())
                    # print(f'history_logit is {history_logit}')
                    history_logit = history_logit[-12:]  # 只保留最新加入的12个元素
                    # print(f'history_logit length is {len(history_logit)}')   # 从1增大到12后就一致保持在12了
                    avg_logit = sum(history_logit)  # 对这27个动作类别的12次结果求和
                    # print(f'avg_logit is {avg_logit}')
                    idx_ = np.argmax(avg_logit, axis=1)[0]
                    # print(f'idx_ is {idx_}')

                idx, idx_history = process_output(idx_, idx_history)

                t2 = time.time()
                print(f"{index} frame, recognition result is {catigories[idx]}")

                spend_time = t2 - t1

            img = cv2.resize(img, (640, 480))
            img = img[:, ::-1]
            height, width, _ = img.shape
            label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

            cv2.putText(label, 'Prediction: ' + catigories[idx],
                        (0, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)
            cv2.putText(label, '{:.1f} Vid/s'.format(1 / spend_time),
                        (width - 170, int(height / 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2)

            img = np.concatenate((img, label), axis=0)
            cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # exit
                break
            elif key == ord('F') or key == ord('f'):  # full screen
                print('Changing full screen option!')
                full_screen = not full_screen
                if full_screen:
                    print('Setting FS!!!')
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)
            if t is None:
                t = time.time()
            else:
                nt = time.time()
                index += 1
                t = nt

        cap.release()
        cv2.destroyAllWindows()


main()
