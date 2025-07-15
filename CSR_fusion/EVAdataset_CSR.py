
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
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from cvtransforms import *
import json
import editdistance


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
    letters = ['我', '饿', '了', '口', '渴', '吃', '饱', '水', '太', '烫', '累', '想', '睡', '觉', '要', '休', '息', '扶', '起', '来', '上', '厕', '所', '坐', '下', '零', '食', '失', '禁', '有', '点', '冷', '好', '热', '闷', '风', '大', '打', '开', '空', '调', '关', '闭', '高', '温', '度', '低', '饭', '喝', '饮', '料', '不', '很', '咬', '动', '床', '果', '生', '病', '紧', '急', '呼', '救', '该', '药', '摔', '倒', '行', '血', '压', '头', '晕', '嗓', '子', '疼', '脖', '腰', '痛', '肩', '膀', '腿', '牙', '齿', '感', '冒', '发', '烧', '窗', '户', '吸', '困', '难', '眼', '睛', '受', '把', '门', '灯', '多', '久', '能', '治', '需', '住', '院', '吗', '这', '效', '节', '用', '心', '跳', '快', '一', '直', '咳', '嗽', '胸', '喘', '气', '情', '况', '严', '重', '手', '术', '传', '染', '全', '身', '乏', '力', '电', '话', '短', '信', '聊', '视', '频', '谢', '你', '客', '没', '听', '清', '是', '样', '的', '楚', '孤', '独', '系', '对', '提', '醒', '帮', '定', '闹', '钟', '剪', '洗', '澡', '锻', '炼', '换', '衣', '服', '按', '摩', '指', '甲', '看', '到', '书', '去', '运', '玩', '游', '戏', '棋', '网', '散', '步', '音', '乐', '往', '前', '走', '停', '向', '左', '转', '右', ' ']
    letters_attn = ['我', '饿', '了', '口', '渴', '吃', '饱', '水', '太', '烫', '累', '想', '睡', '觉', '要', '休', '息', '扶', '起', '来', '上', '厕', '所', '坐', '下', '零', '食', '失', '禁', '有', '点', '冷', '好', '热', '闷', '风', '大', '打', '开', '空', '调', '关', '闭', '高', '温', '度', '低', '饭', '喝', '饮', '料', '不', '很', '咬', '动', '床', '果', '生', '病', '紧', '急', '呼', '救', '该', '药', '摔', '倒', '行', '血', '压', '头', '晕', '嗓', '子', '疼', '脖', '腰', '痛', '肩', '膀', '腿', '牙', '齿', '感', '冒', '发', '烧', '窗', '户', '吸', '困', '难', '眼', '睛', '受', '把', '门', '灯', '多', '久', '能', '治', '需', '住', '院', '吗', '这', '效', '节', '用', '心', '跳', '快', '一', '直', '咳', '嗽', '胸', '喘', '气', '情', '况', '严', '重', '手', '术', '传', '染', '全', '身', '乏', '力', '电', '话', '短', '信', '聊', '视', '频', '谢', '你', '客', '没', '听', '清', '是', '样', '的', '楚', '孤', '独', '系', '对', '提', '醒', '帮', '定', '闹', '钟', '剪', '洗', '澡', '锻', '炼', '换', '衣', '服', '按', '摩', '指', '甲', '看', '到', '书', '去', '运', '玩', '游', '戏', '棋', '网', '散', '步', '音', '乐', '往', '前', '走', '停', '向', '左', '转', '右', ' ', '<BOS>', '<EOS>']
    filename = "/ai/mm/corpus.json"
    with open (filename, 'r', encoding='utf-8') as f_obj:
        corpus = json.load(f_obj)
    command = list(corpus.values())

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
    
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
    
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):  
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer


    def __init__(self, set, directory):
        self.set = set
        self.file_list = self.build_file_list(set, directory)

        print('Total num of samples: ', len(self.file_list))

    
    def _load_anno(self, label):
        lines = MyDataset.command[label]
        return MyDataset.txt2arr(lines, 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
        

    def __getitem__(self, idx):


        audio, fs = librosa.load(self.file_list[idx][2])
        audio = add_noise(audio)
        audio = Audio_MFSC(fs,audio)

        emg = sio.loadmat(self.file_list[idx][3])
        emg = np.expand_dims(emg["data"], axis=0)
        emg = filter(emg)
        emg = EMG_MFSC(emg)

        lip = getData(self.file_list[idx][1])
        lip = lip.reshape(60,88,88,1)
        lip = np.rollaxis(lip, 3, 1)

        label = int(self.file_list[idx][0])
        anno = self._load_anno(label)
        anno_len = anno.shape[0]
        anno_attn = np.insert(anno, 0, MyDataset.letters_attn.index('<BOS>') + 1)
        anno_attn = np.append(anno_attn, MyDataset.letters_attn.index('<EOS>') + 1)
        audio = torch.FloatTensor(audio[np.newaxis, :])
        emg = torch.FloatTensor(emg)
        lip = torch.FloatTensor(lip)
        anno_pad = self._padding(anno, 5)
        anno_attn_pad = self._padding(anno_attn, 7) ## max_length = 5 + 2
        fusion_len = 92

        return audio, emg, lip, torch.LongTensor(anno_pad), anno_len, fusion_len, torch.LongTensor(anno_attn_pad)

    def __len__(self):
        return len(self.file_list)



