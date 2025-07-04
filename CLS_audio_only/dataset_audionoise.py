
# encoding: utf-8
import os
import glob
import random
import numpy as np
import librosa
import torch
from python_speech_features import mfcc
# from lpctorch import LPCCoefficients
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt


def add_noise(signal):
    SNR = -10
    noise = np.random.randn(signal.shape[0]) # 高斯白噪声
    noise = noise - np.mean(noise)
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal
    return signal_noise


def get_MFSC(fs, x):
    x = np.array(x).astype(np.float)
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


        audio_subject_list = np.load('audio_subj_list.npy')
        audio_subject_list = audio_subject_list.tolist()

        audio_test_dataset = audio_subject_list[0:8]
        del audio_subject_list[0:8]
        audio_train_dataset = audio_subject_list

        for i in range(32):
            audio_dataset = audio_train_dataset[i]
            sessions = os.listdir(audio_dataset)
            for session in sessions:
                samples = os.listdir(audio_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    audiopath = audio_dataset + '/' + session + '/' + sample
                    entry = (label, audiopath)
                    trnList.append(entry)

        for i in range(8):
            audio_dataset = audio_test_dataset[i]
            sessions = os.listdir(audio_dataset)
            for session in sessions:
                samples = os.listdir(audio_dataset + '/' + str(session))
                for sample in samples:
                    label = sample.split('.')[0]
                    audiopath = audio_dataset + '/' + session + '/' + sample
                    entry = (label, audiopath)
                    tstList.append(entry)


        random.shuffle(trnList)
        random.shuffle(tstList)

        valList = trnList[:2020]
        random.shuffle(valList)

        if set == 'train':
            return trnList
        if set == 'val':
            return valList
        if set == 'test':
            return tstList

        # completeList : 列表保存 ( label ，文件绝对路径 )

    def __init__(self, set, directory):
        self.set = set
        # file_list : 文件列表
        self.file_list = self.build_file_list(set, directory)

        # 打印该类型数据集（ 训练 or 测试 ）的样本总数
        print('Total num of samples: ', len(self.file_list))
        

    def __getitem__(self, idx):

        audio, fs = librosa.load(self.file_list[idx][1])
        audio = add_noise(audio)
        # audio = mfcc(audio)
        audio = get_MFSC(fs,audio)
        # audio = LPCCoefficients(audio)
        # print('sampling rate:', fs) fs = 22050


        label = int(self.file_list[idx][0])
        audio = torch.FloatTensor(audio[np.newaxis, :])

        return audio, label

    def __len__(self):
        return len(self.file_list)