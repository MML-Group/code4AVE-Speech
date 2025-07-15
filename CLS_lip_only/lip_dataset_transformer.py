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
    def build_file_list(self, set, dir):
        trnList = []
        valList = []
        tstList = []

        lip_subject_list = np.load('/ai/exp2/fusion_baseline_231205_new/lip_subject.npy')
        lip_subject_list = lip_subject_list.tolist()

        # training dataset
        for i in range(70):
            lip_dataset = lip_subject_list[i]
            sessions = os.listdir(lip_dataset)
            for session in sessions:
                samples = os.listdir(lip_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    entry = (label, videopath)
                    trnList.append(entry)


        # validation dataset
        for i in range(10):
            lip_dataset = lip_subject_list[i+70]
            sessions = os.listdir(lip_dataset)
            for session in sessions:
                samples = os.listdir(lip_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    entry = (label, videopath)
                    valList.append(entry)


        # testing dataset
        for i in range(20):
            lip_dataset = lip_subject_list[i+80]
            sessions = os.listdir(lip_dataset)
            for session in sessions:
                samples = os.listdir(lip_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    entry = (label, videopath)
                    tstList.append(entry)


        random.shuffle(trnList)
        random.shuffle(tstList)
        random.shuffle(valList)


        if set == 'train':
            return trnList
        if set == 'val':
            return valList
        if set == 'test':
            return tstList

    def __init__(self, set, directory):
        self.set = set
        self.file_list = self.build_file_list(set, directory)

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
