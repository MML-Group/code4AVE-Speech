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
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from cvtransforms import *


def filter(raw_data):
    fs=1000
    b1, a1 = signal.iirnotch(50, 30, fs)    # second-order Butterworth notch filter
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass')  # Butterworth band-pass filter

    x = signal.filtfilt(b1, a1, raw_data, axis=1)
    x = signal.filtfilt(b2, a2, x, axis=1)
    x = signal.filtfilt(b3, a3, x, axis=1)
    x = signal.filtfilt(b4, a4, x, axis=1)
    x = signal.filtfilt(b5, a5, x, axis=1)
    return x

def EMG_MFSC(x):
    x = x[:,250:,:]
    n_mels = 36
    sr = 1000
    channel_list = []
    for j in range(x.shape[-1]):                             # channels
        mfsc_x = np.zeros((x.shape[0], 36, n_mels))
        for i in range(x.shape[0]):                          # samples
#             norm_x = x[i, :, j]/np.max(abs(x[i, :, j]))
            norm_x = np.asfortranarray(x[i, :, j])
            tmp = librosa.feature.melspectrogram(y=norm_x, sr=sr, n_mels=n_mels, n_fft=200, hop_length=50)
            tmp = librosa.power_to_db(tmp).T
            mfsc_x[i, :, :] = tmp

        mfsc_x = np.expand_dims(mfsc_x, axis=-1)
        channel_list.append(mfsc_x)
    data_x = np.concatenate(channel_list, axis=-1)
    mu = np.mean(data_x)
    std = np.std(data_x)
    data_x = (data_x - mu) / std
    data_x = data_x.transpose(0,3,1,2)


    return data_x # ()


class MyDataset():
    def build_file_list(self, set, dir):
        trnList = []
        valList = []
        tstList = []

        emg_subject_list = np.load('/ai/exp2/fusion_baseline_231205_new/emg_subject.npy')
        emg_subject_list = emg_subject_list.tolist()

        # training dataset
        for i in range(70):
            emg_dataset = emg_subject_list[i]
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    entry = (label, emgpath)
                    trnList.append(entry)


        # validation dataset
        for i in range(10):
            emg_dataset = emg_subject_list[i+70]
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    entry = (label, emgpath)
                    valList.append(entry)


        # testing dataset
        for i in range(20):
            emg_dataset = emg_subject_list[i+80]
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    entry = (label, emgpath)
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
        emg = sio.loadmat(self.file_list[idx][1])
        emg = np.expand_dims(emg["data"], axis=0)
        emg = filter(emg)
        emg = EMG_MFSC(emg)



        label = int(self.file_list[idx][0])
        emg = torch.FloatTensor(emg)


        return emg, label

    def __len__(self):
        return len(self.file_list)
