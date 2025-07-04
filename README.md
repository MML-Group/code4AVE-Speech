# AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals

![Multi-modal speech recognition](figures/multi-modal-speech-recognition.png)

## Abstract

The global aging population faces considerable challenges, particularly in communication, due to the prevalence of hearing and speech impairments. To address these, we introduce the AVE speech, a comprehensive multi-modal dataset for speech recognition tasks. The dataset includes a 100-sentence Mandarin corpus with audio signals, lip-region video recordings, and six-channel electromyography (EMG) data, collected from 100 participants. Each subject read the entire corpus ten times, with each sentence averaging approximately two seconds in duration, resulting in over 55 hours of multi-modal speech data per modality. Experiments demonstrate that combining these modalities significantly improves recognition performance, particularly in cross-subject and high-noise environments. To our knowledge, this is the first publicly available sentence-level dataset integrating these three modalities for large-scale Mandarin speech recognition. We expect this dataset to drive advancements in both acoustic and non-acoustic speech recognition research, enhancing cross-modal learning and human-machine interaction.

## AVE Speech Dataset – Source Code

This repository contains the source code for *AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals*.

The dataset is available at:
👉 [AVE-Speech Dataset on Hugging Face](https://huggingface.co/datasets/MML-Group/AVE-Speech)

## Implementation
Included codelines can be used for two speech recognition tasks, i.e., word-level continuous speech recognition (CSR) and sentence-level speech classification (CLS). 

Dataset preparation steps for mentioned tasks can be found in the CLS_fusion and CSR_fusion folders, and corresponding changes can be made to fulfill particular requirements upon your settings.  

## Citation
If you use the source code in your work, please cite it as:
```
@article{zhou2025ave,
  title={AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals},
  author={Zhou, Dongliang and Zhang, Yakun and Wu, Jinghan and Zhang, Xingyu and Xie, Liang and Yin, Erwei},
  journal={IEEE Transactions on Human-Machine Systems},
  year={2025}
}
```

