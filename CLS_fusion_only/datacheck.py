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


audio_subject_list = np.load('audio_subject.npy')
audio_subject_list = audio_subject_list.tolist()

lip_subject_list = np.load('lip_subject.npy')
lip_subject_list = lip_subject_list.tolist()

emg_subject_list = np.load('emg_subject.npy')
emg_subject_list = emg_subject_list.tolist()

trnList = []
valList = []
tstList = []

# training dataset
for i in range(70):
    audio_dataset = audio_subject_list[i]
    lip_dataset = lip_subject_list[i]
    emg_dataset = emg_subject_list[i]
    sessions = os.listdir(emg_dataset)
    for session in sessions:
        samples = os.listdir(emg_dataset + '/' + str(session))
        for sample in samples:
            label = sample.split('.')[0]
            emgpath = emg_dataset + '/' + session + '/' + sample
            if os.path.exists(lip_dataset + '/' + session + '/' + str(label) + '.avi'):
                videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'lippath missing')
            if os.path.exists(audio_dataset + '/' + session + '/' + str(label) + '.wav'):
                audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'audio path missing')
            entry = (label, videopath, audiopath, emgpath)
            trnList.append(entry)


# validation dataset
for i in range(10):
    audio_dataset = audio_subject_list[i+70]
    lip_dataset = lip_subject_list[i+70]
    emg_dataset = emg_subject_list[i+70]
    sessions = os.listdir(emg_dataset)
    for session in sessions:
        samples = os.listdir(emg_dataset + '/' + str(session))
        for sample in samples:
            label = sample.split('.')[0]
            emgpath = emg_dataset + '/' + session + '/' + sample
            if os.path.exists(lip_dataset + '/' + session + '/' + str(label) + '.avi'):
                videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'lippath missing')
            if os.path.exists(audio_dataset + '/' + session + '/' + str(label) + '.wav'):
                audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'audio path missing')
            entry = (label, videopath, audiopath, emgpath)
            valList.append(entry)


# testing dataset
for i in range(20):
    audio_dataset = audio_subject_list[i+80]
    lip_dataset = lip_subject_list[i+80]
    emg_dataset = emg_subject_list[i+80]
    sessions = os.listdir(emg_dataset)
    for session in sessions:
        samples = os.listdir(emg_dataset + '/' + str(session))
        for sample in samples:
            label = sample.split('.')[0]
            emgpath = emg_dataset + '/' + session + '/' + sample
            if os.path.exists(lip_dataset + '/' + session + '/' + str(label) + '.avi'):
                videopath = lip_dataset + '/' + session + '/' + str(label) + '.avi'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'lippath missing')
            if os.path.exists(audio_dataset + '/' + session + '/' + str(label) + '.wav'):
                audiopath = audio_dataset + '/' + session + '/' + str(label) + '.wav'
            else:
                print(emg_dataset + '/' + session + '/' + sample, 'audio path missing')
            entry = (label, videopath, audiopath, emgpath)
            tstList.append(entry)


print(len(trnList), len(valList), len(tstList),)