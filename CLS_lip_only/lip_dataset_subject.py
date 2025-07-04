# encoding: utf-8
import os
import glob
import scipy.io as sio
import random
import numpy as np
import librosa
from scipy import signal
import torch
import cv2
from python_speech_features import mfcc
# from lpctorch import LPCCoefficients
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from lip_cvtransforms import *

def getData(path):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(5))
    num_frames = 60
    list = []
    for num in range(num_frames):
        if num%1 == 0:
            ret , frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                frame = CenterCrop(frame,(320,240))
                frame = cv2.resize(frame,(88,88))

                
            else:
                frame = np.zeros((88,88))
            list.append(frame)
    arrays = np.array(list)
    arrays = arrays / 255.


    return arrays


class MyDataset():
    def build_file_list(self, subj):
        List = []

        lip_subject_list = np.load('/ai/benchmark/fusion_baseline_231205_new/lip_subject.npy')
        lip_subject_list = lip_subject_list.tolist()

        lip_dataset = lip_subject_list[subj]
        sessions = os.listdir(lip_dataset)
        for session in sessions:
            samples = os.listdir(lip_dataset + '/' + str(session))
            for sample in samples:
                label = sample.split('.')[0]
                videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                entry = (label, videopath)
                List.append(entry)


        random.shuffle(List)
        return List
        # completeList : 列表保存 ( label ，文件绝对路径 )

    def __init__(self, subject):
        self.file_list = self.build_file_list(subject)

        # 打印该类型数据集（ 训练 or 测试 ）的样本总数
        print('Total num of samples: ', len(self.file_list))
        

    def __getitem__(self, idx):


        lip = getData(self.file_list[idx][1])
        lip = lip.reshape(60,88,88,1)
        lip = np.rollaxis(lip, 3, 1)


        label = int(self.file_list[idx][0])
        lip = torch.FloatTensor(lip)

        return lip, label

    def __len__(self):
        return len(self.file_list)