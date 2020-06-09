#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
from random import sample

#default args:
default_args = {'noise_prob': 0.1,
                'gasuss_mean': 0,
                'gasuss_var': 0.001,
                'rand_hug': 30,
                'rand_saturation':30,
                'rand_light': 30,
                'rot_angle': 15,
                'bordervalue': (127, 127, 127),
                'zoom_out_value': 0.7,
                'blur_kernel': (7, 7),
                'output_shape': (608, 608),
                'take_value' : 5
               }

#添加黑色noise
def sp_noise(image, box_loc=None, **kwargs):
    h, w = image.shape[0:2]
    noise = np.random.rand(h,w)
    out_img = image.copy()
    out_img[noise < kwargs['noise_prob']] = 0
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#高斯noise
def gasuss_noise(image, box_loc=None, **kwargs):
    out_img = (image / 255.) - 0.5
    noise = np.random.normal(kwargs['gasuss_mean'], kwargs['gasuss_var']** 0.5, image.shape)
    out_img = out_img + noise + 0.5
    out_img[out_img < 0] = 0
    out_img[out_img > 1] = 1
    out_img = (out_img * 255).astype(np.uint8)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#gaussian blur
def blur(image, box_loc=None, **kwargs):
    out_img = cv2.GaussianBlur(image, kwargs['blur_kernel'], 0)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整彩度(彩度通道加上隨機-N~N之值)
def mod_hue(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,0] += np.random.randint(-kwargs['rand_hug'], kwargs['rand_hug'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 180).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整飽和度(飽和度通道加上隨機-N~N之值)
def mod_saturation(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,1] += np.random.randint(-kwargs['rand_saturation'], kwargs['rand_saturation'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#調整亮度(亮度通道加上隨機-N~N之值)
def mod_light(image, box_loc=None, **kwargs):
    out_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    out_img[:,:,2] += np.random.randint(-kwargs['rand_light'], kwargs['rand_light'])
    out_img = cv2.cvtColor(np.clip(out_img, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if box_loc is None:
        return out_img
    else:
        return out_img, box_loc

#水平翻轉
def horizontal_flip(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(x_min, y_min, x_max, y_max, label)
    '''
    if box_loc is None:
        return cv2.flip(image, 1)
    else:
        w = image.shape[1]
        for i in box_loc:
            if i[2] == 0:
                break
            else:
                x_min, x_max = i[0], i[2]
                i[0] = w - x_max
                i[2] = w - x_min
        return cv2.flip(image, 1), box_loc

#垂直翻轉
def vertical_flip(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(num box,(x_min, y_min, x_max, y_max, label))
    '''
    if box_loc is None:
        return cv2.flip(image, 0)
    else:
        h = image.shape[0]
        for i in box_loc:
            if i[3] == 0:
                break
            else:
                y_min, y_max = i[1], i[3]
                i[1] = h - y_max
                i[3] = h - y_min
        return cv2.flip(image, 0), box_loc

#旋轉-n~n度
def rot_image(image, box_loc=None, **kwargs):
    '''
    Args:
        box_loc: bounding box location(num box,(x_min, y_min, x_max, y_max, label))
        rot: 要選轉的範圍
        bordervalue: 空白處補的值
    '''
    h, w, _ = image.shape
    center = ( w // 2, h // 2)
    angle = np.random.randint(-kwargs['rot_angle'], kwargs['rot_angle'])
    M = cv2.getRotationMatrix2D(center, angle, 1)
    out_img = cv2.warpAffine(image, M, (w, h), borderValue = kwargs['bordervalue'])
    if box_loc is None:
        return out_img
    else:
        loc = box_loc[:,0:4].copy()
        loc = np.append(loc, loc[:, 0:1], axis=-1)
        loc = np.append(loc, loc[:, 3:4], axis=-1)
        loc = np.append(loc, loc[:, 2:3], axis=-1)
        loc = np.append(loc, loc[:, 1:2], axis=-1)
        loc = loc.reshape(-1, 4, 2)
        loc = loc - np.array(center)
        rot_loc = loc.dot(np.transpose(M[:,0:2]))
        rot_loc = rot_loc + np.array(center)
        rot_box = np.hstack([np.min(rot_loc, axis=-2), np.max(rot_loc, axis=-2), box_loc[:, 4:5]])
        rot_box = np.floor(rot_box)
        rot_box[...,0:4] = np.clip(rot_box[...,0:4], [0,0,0,0], [w-1, h-1, w-1, h-1])

        return out_img, rot_box

#等比例縮放影像
def resize_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape
    max_edge = max(kwargs['output_shape'][0], kwargs['output_shape'][1])
    scale = min( max_edge / h, max_edge / w)
    h = int(h * scale)
    w = int(w * scale)

    if box_loc is None:
        return cv2.resize(image, (w, h))
    else:
        box_loc[:,0] = box_loc[:,0] * scale
        box_loc[:,1] = box_loc[:,1] * scale
        box_loc[:,2] = box_loc[:,2] * scale
        box_loc[:,3] = box_loc[:,3] * scale
        return cv2.resize(image, (w, h)), box_loc.astype(np.int32)

#將樸片補至指定大小
def padding_img(image, box_loc=None, **kwargs):
    h, w, _ = image.shape

    dx = int((kwargs['output_shape'][1] - w) / 2)
    dy = int((kwargs['output_shape'][0] - h) / 2)

    out_img = np.ones((kwargs['output_shape'][0], kwargs['output_shape'][1], 3), np.uint8) * kwargs['bordervalue'][0]
    out_img[dy: dy + h, dx: dx + w, :] = cv2.resize(image, (w, h))

    if box_loc is None:
        return out_img
    else:
        box_loc[:,0] = box_loc[:,0] + dx
        box_loc[:,1] = box_loc[:,1] + dy
        box_loc[:,2] = box_loc[:,2] + dx
        box_loc[:,3] = box_loc[:,3] + dy
        return out_img, box_loc.astype(np.int32)

#隨機縮小 value~1倍
def random_zoom_out(image, box_loc=None, **kwargs):

    h, w, _ = image.shape
    scale = np.random.uniform(kwargs['zoom_out_value'], 1)
    h = int(h * scale)
    w = int(w * scale)
    dx = int((image.shape[1] - w) / 2)
    dy = int((image.shape[0] - h) / 2)
    out_img = np.ones(image.shape, np.uint8) * kwargs['bordervalue'][0]
    out_img[dy: dy + h, dx: dx + w] = cv2.resize(image, (w, h))

    if box_loc is None:
        return out_img
    else:
        box_loc[:,0] = box_loc[:,0] * scale + dx
        box_loc[:,1] = box_loc[:,1] * scale + dy
        box_loc[:,2] = box_loc[:,2] * scale + dx
        box_loc[:,3] = box_loc[:,3] * scale + dy
        return out_img, box_loc.astype(np.int32)

        
#隨機選擇0~N個 image augmentation方法
def rand_aug_image(image, box_loc=None, **kwargs):

    if box_loc is None:
        out_img = resize_img(image, **kwargs)
    else:
        out_img, box_loc = resize_img(image, box_loc, **kwargs)

    #total augmentation function
    func_list = [sp_noise, gasuss_noise, mod_hue, mod_saturation, mod_light,
                 horizontal_flip, vertical_flip, rot_image, random_zoom_out,
                 blur]
    
    #rand take function
    take_func = sample(func_list, np.random.randint(kwargs['take_value']))

    for func in take_func:
        if box_loc is None:
            out_img = func(out_img, **kwargs)
        else:
            out_img, box_loc = func(out_img, box_loc, **kwargs)
    if box_loc is None:
        out_img = padding_img(out_img, **kwargs)
        return out_img
    else:
        out_img, box_loc = padding_img(out_img, box_loc, **kwargs)
        return out_img, box_loc