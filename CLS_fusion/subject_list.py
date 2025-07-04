import os
import numpy as np

dataset = '/ai/memory/wujinghan/audiodata'
#dataset = "E:/TAIIC/audioonly/audio/audio"
subject_list = []
# dataset = 'E:/TAIIC/养老助残数据集/audio/audio_crosssubj_test'
# subject_list = np.load('subj_list.npy')
# subject_list = subject_list.tolist()
names = os.listdir(dataset)
for name in names:
    subject_list.append(dataset + '/' + name)


subject_list.sort(key=lambda x:int(x.split('_')[1]))
np.save('audio_subject.npy', subject_list)
# print(subject_list)
print(len(subject_list))

# subjlist = np.load('lip_subj_list.npy')
# print(subjlist)

# subj_list = np.load('emg_subject.npy')
# subj_list = subj_list.tolist()
# trnList = []
# valList = []
# tstList = []

# root_path = '/ai/multitask2/wujinghan/'
# for i in range(70):
#     subj_path = subj_list[i]
#     subject_path = root_path + subj_path.split('/')[3]
#     sessions = os.listdir(subject_path)
#     for session in sessions:
#         samples = os.listdir(subject_path + '/' + str(session))
#         for sample in samples:
#             emgpath = subject_path + '/' + session + '/' + sample
#             trnList.append(emgpath)


# for i in range(10):
#     subj_path = subj_list[i+70]
#     subject_path = root_path + subj_path.split('/')[3]
#     sessions = os.listdir(subject_path)
#     for session in sessions:
#         samples = os.listdir(subject_path + '/' + str(session))
#         for sample in samples:
#             emgpath = subject_path + '/' + session + '/' + sample
#             valList.append(emgpath)

# for i in range(20):
#     subj_path = subj_list[i+80]
#     subject_path = root_path + subj_path.split('/')[3]
#     sessions = os.listdir(subject_path)
#     for session in sessions:
#         samples = os.listdir(subject_path + '/' + str(session))
#         for sample in samples:
#             emgpath = subject_path + '/' + session + '/' + sample
#             tstList.append(emgpath)


# print('trnlist: ', len(trnList))
# print('vallist: ', len(valList))
# print('testlist: ', len(tstList))