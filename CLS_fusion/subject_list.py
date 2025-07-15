import os
import numpy as np

## create the subject list file before data processing

dataset = 'audiodata' # path to the audio data
subject_list = []
names = os.listdir(dataset)
for name in names:
    subject_list.append(dataset + '/' + name)


subject_list.sort(key=lambda x:int(x.split('_')[1]))
np.save('audio_subject.npy', subject_list)
# print(subject_list)
print(len(subject_list))

