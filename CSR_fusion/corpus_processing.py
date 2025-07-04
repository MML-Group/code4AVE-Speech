
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
import json

filename = "corpus.json"
with open (filename, 'r', encoding='utf-8') as f_obj:
    corpus = json.load(f_obj)
command = list(corpus.values())

diction = []
# diction = ['我', '饿', '了', '口', '渴', '吃', '饱', '水', '太', '烫', '累', '想', '睡', '觉', '要', '休', '息', '扶', '起', '来', '上', '厕', '所', '坐', '下', '零', '食', '失', '禁', '有', '点', '冷', '好', '热', '闷', '风', '大', '打', '开', '空', '调', '关', '闭', '高', '温', '度', '低', '饭', '喝', '饮', '料', '不', '很', '咬', '动', '床', '果', '生', '病', '紧', '急', '呼', '救', '该', '药', '摔', '倒', '行', '血', '压', '头', '晕', '嗓', '子', '疼', '脖', '腰', '痛', '肩', '膀', '腿', '牙', '齿', '感', '冒', '发', '烧', '窗', '户', '吸', '困', '难', '眼', '睛', '受', '把', '门', '灯', '多', '久', '能', '治', '需', '住', '院', '吗', '这', '效', '节', '用', '心', '跳', '快', '一', '直', '咳', '嗽', '胸', '喘', '气', '情', '况', '严', '重', '手', '术', '传', '染', '全', '身', '乏', '力', '电', '话', '短', '信', '聊', '视', '频', '谢', '你', '客', '没', '听', '清', '是', '样', '的', '楚', '孤', '独', '系', '对', '提', '醒', '帮', '定', '闹', '钟', '剪', '洗', '澡', '锻', '炼', '换', '衣', '服', '按', '摩', '指', '甲', '看', '到', '书', '去', '运', '玩', '游', '戏', '棋', '网', '散', '步', '音', '乐', '往', '前', '走', '停', '向', '左', '转', '右', ' ']

for i in range(101):
    line = command[i]
    words = list(line)
    for word in words:
        if word not in diction:
            diction.append(word)

print(diction)  ## len(diction) = 194, including ' '
