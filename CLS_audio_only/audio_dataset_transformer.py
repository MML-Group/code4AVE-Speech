
# encoding: utf-8
import os
import glob
import random
import numpy as np
import librosa
import torch
from python_speech_features import mfcc
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt


def add_noise(signal):
    SNR = -10
    noise = np.random.randn(signal.shape[0]) 
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise


def get_MFSC(fs, x):
    x = np.array(x).astype(float)
    n_mels = 64
    norm_x = x[:]
    tmp = librosa.feature.melspectrogram(y=norm_x, sr=fs, n_mels=n_mels, n_fft=1500, hop_length=735)
    mfsc_x = librosa.power_to_db(tmp).T
    return mfsc_x  # (60,64)


class MyDataset():
    def build_file_list(self, set, dir):
        trnList = []
        valList = []
        tstList = []


        audio_subject_list = np.load('/ai/exp3/fusion_baseline_231205_new/audio_only/audio_subject.npy')
        audio_subject_list = audio_subject_list.tolist()
        

        # training dataset
        for i in range(70):
            audio_dataset = audio_subject_list[i]
            sessions = os.listdir(audio_dataset)
            for session in sessions:
                samples = os.listdir(audio_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, audiopath)
                    trnList.append(entry)


        # validation dataset
        for i in range(10):
            audio_dataset = audio_subject_list[i+70]
            sessions = os.listdir(audio_dataset)
            for session in sessions:
                samples = os.listdir(audio_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, audiopath)
                    valList.append(entry)


        # testing dataset
        for i in range(20):
            audio_dataset = audio_subject_list[i+80]
            sessions = os.listdir(audio_dataset)
            for session in sessions:
                samples = os.listdir(audio_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
                    entry = (label, audiopath)
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

        audio, fs = librosa.load(self.file_list[idx][1])
        audio = add_noise(audio)
        audio = get_MFSC(fs,audio)


        label = int(self.file_list[idx][0])
        audio = torch.FloatTensor(audio[np.newaxis, :])

        return audio, label

    def __len__(self):
        return len(self.file_list)
