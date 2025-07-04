# coding: utf-8
import random
import cv2
import numpy as np


def CenterCrop(batch_img, size):
    w, h = batch_img.shape[1], batch_img.shape[0]  #(w=640, h=360)
    tw, th = size
    img = np.zeros((th, tw))
    x1 = int((w - tw)/2.)
    y1 = int((h - th)/2.)    
    img = batch_img[y1:y1+th, x1:x1+tw]

    return img


def HorizontalFlip(batch_img):
    for i in range(len(batch_img)):
        if random.random() > 0.5:
            for j in range(len(batch_img[i])):
                batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
    return batch_img

def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img

# def RandomCrop(batch_img, size):
#     w, h = batch_img.shape[1], batch_img.shape[0]
#     tw, th = size
#     img = np.zeros((th, tw))
#     # x1 = random.randint(0, 8)
#     # y1 = random.randint(0, 8)
#     x0 = int((w - tw)/2.)
#     y0 = int((h - th)/2.)  
#     # x1 = random.randint(x0, w/2)
#     # y1 = random.randint(y0, h/2)
#     x1 = random.randint(x0, (w - tw))
#     y1 = random.randint(y0, (h - th))
#     img = batch_img[y1:y1+th,x1:x1+tw]
#     return img

def RandomCrop(batch_img, size):
    w, h = batch_img.shape[1], batch_img.shape[0]
    th, tw = size
    # img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
    img = np.zeros((th, tw))
    # for i in range(len(batch_img)):
    x1 = random.randint(80, 160)
    y1 = random.randint(30, 60)
    img = batch_img[y1:y1+th, x1:x1+tw]
    return img