
# encoding: utf-8
import os
import glob
from re import sub
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
from cvtransforms import *


# audio data preprocessing

def add_noise(signal):
    SNR = -10
    noise = np.random.randn(signal.shape[0])
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise


def Audio_MFSC(fs, x):

    x = np.array(x).astype(float)
    n_mels = 64
    norm_x = x[:]
    tmp = librosa.feature.melspectrogram(y=norm_x, sr=fs, n_mels=n_mels, n_fft=1500, hop_length=735)
    mfsc_x = librosa.power_to_db(tmp).T
    return mfsc_x  # (60,64)


# emg data preprocessing

def filter(raw_data):
    fs=1000
    b1, a1 = signal.iirnotch(50, 30, fs) 
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10/(fs/2), 400/(fs/2)], 'bandpass') 

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
    for j in range(x.shape[-1]):                            
        mfsc_x = np.zeros((x.shape[0], 36, n_mels))
        for i in range(x.shape[0]):                         
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


    return data_x 



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


        audio_subject_list = np.load('/ai/mm/audio_subject.npy')
        audio_subject_list = audio_subject_list.tolist()


        lip_subject_list = np.load('/ai/mm/lip_subject.npy')
        lip_subject_list = lip_subject_list.tolist()


        emg_subject_list = np.load('/ai/mm/emg_subject.npy')
        emg_subject_list = emg_subject_list.tolist()


        # training dataset
        for i in range(70):
            audio_dataset = audio_subject_list[i]
            audio_dataset = str(audio_dataset).replace('home', 'ai/memory')
            lip_dataset = lip_subject_list[i]
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
            emg_dataset = emg_subject_list[i]
            emg_dataset = str(emg_dataset).replace('home', 'ai/memory')
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, videopath, audiopath, emgpath)
                    trnList.append(entry)


        # validation dataset
        for i in range(10):
            audio_dataset = audio_subject_list[i+70]
            audio_dataset = str(audio_dataset).replace('home', 'ai/memory')
            lip_dataset = lip_subject_list[i+70]
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
            emg_dataset = emg_subject_list[i+70]
            emg_dataset = str(emg_dataset).replace('home', 'ai/memory')
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, videopath, audiopath, emgpath)
                    valList.append(entry)


        # testing dataset
        for i in range(20):
            audio_dataset = audio_subject_list[i+80]
            audio_dataset = str(audio_dataset).replace('home', 'ai/memory')
            lip_dataset = lip_subject_list[i+80]
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
            emg_dataset = emg_subject_list[i+80]
            emg_dataset = str(emg_dataset).replace('home', 'ai/memory')
            sessions = os.listdir(emg_dataset)
            for session in sessions:
                samples = os.listdir(emg_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    emgpath = emg_dataset + '/' + session + '/' + sample
                    videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, videopath, audiopath, emgpath)
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

        audio, fs = librosa.load(self.file_list[idx][2])
        audio = Audio_MFSC(fs,audio)
        
        emg = sio.loadmat(self.file_list[idx][3])
        emg = np.expand_dims(emg["data"], axis=0)
        emg = filter(emg)
        emg = EMG_MFSC(emg)

        lip = getData(self.file_list[idx][1])
        lip = lip.reshape(60,88,88,1)
        lip = np.rollaxis(lip, 3, 1)



        label = int(self.file_list[idx][0])
        audio = torch.FloatTensor(audio[np.newaxis, :])
        emg = torch.FloatTensor(emg)
        lip = torch.FloatTensor(lip)




        return audio, emg, lip, label

    def __len__(self):
        return len(self.file_list)



