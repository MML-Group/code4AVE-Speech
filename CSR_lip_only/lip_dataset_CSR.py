
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
from lip_cvtransforms import *
import json
import editdistance


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
    filename = "/ai/mm/corpus.json"
    with open (filename, 'r', encoding='utf-8') as f_obj:
        corpus = json.load(f_obj)
    command = list(corpus.values())

    def build_file_list(self, set, dir):
        trnList = []
        valList = []
        tstList = []


        lip_subject_list = np.load('/ai/mm/lip_subject.npy')
        lip_subject_list = lip_subject_list.tolist()

        # training dataset
        for i in range(70):
            lip_dataset = lip_subject_list[i]
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
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
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
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
            lip_dataset = str(lip_dataset).replace('home', 'ai/memory')
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


        lip = getData(self.file_list[idx][1])
        lip = lip.reshape(60,88,88,1)
        lip = np.rollaxis(lip, 3, 1)

        label = int(self.file_list[idx][0])
        anno = self._load_anno(label)
        lip = torch.FloatTensor(lip)
        lip_len = 60
        anno_len = anno.shape[0]
        anno_pad = self._padding(anno, 5)





        return lip, torch.LongTensor(anno_pad), anno_len, lip_len

    def __len__(self):
        return len(self.file_list)



