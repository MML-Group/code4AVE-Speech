# AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals

![Multi-modal speech recognition](figures/multi-modal-speech-recognition.png)

## Abstract

AVE Speech is a large-scale Mandarin speech corpus that pairs synchronized audio, lip video and surface electromyography (EMG) recordings. The dataset contains 100 sentences read by 100 native speakers. Each participant repeated the full corpus ten times, yielding over **55 hours** of data per modality. These complementary signals enable research on robust acoustic and non-acoustic speech recognition.

## Dataset
- **Modality coverage**: 16 kHz audio, 60x88x88 grayscale lip videos and six-channel EMG signals sampled at 1 kHz.
- **Download**: 👉 [AVE-Speech dataset on Hugging Face](https://huggingface.co/datasets/MML-Group/AVE-Speech).
- **Structure**: recordings are organised by subject and session, e.g.
```
dataset_root/phase
├─ subject_01/
│  ├─ session_01/
│  │  ├─ 0001.wav    # audio
│  │  ├─ 0001.avi    # lip video
│  │  └─ 0001.mat    # EMG
│  └─ ...
```

## Repository Layout
- `CLS_*` – sentence-level speech classification models for individual modalities and their fusion.
- `CSR_*` – word-level continuous speech recognition models.
- `pretrained-models/` – checkpoints for both tasks.
 

## Environment Setup
```bash
conda create -n ave_speech python=3.8 -y
conda activate ave_speech
pip install -r requirements.txt
```
Clone this repository after activating the environment:
```bash
git clone https://github.com/MML-Group/code4AVE-Speech.git
```

## Sentence-Level Classification (CLS)
Run any of the modality-specific or fusion models using the provided scripts. Example for the audio-only baseline:
```bash
python CLS_audio_only/main.py --dataset /path/to/dataset_root --batch-size 36 --epochs 20
```
Replace `CLS_audio_only` with `CLS_emg_only`, `CLS_lip_only`, or `CLS_fusion` to train other variants. Pre-trained models can be loaded using `--path`.

## Word-Level Continuous Speech Recognition (CSR)
The continuous recognition models follow a similar interface. For instance, to train the multi-modal fusion model:
```bash
python CSR_fusion/main_fusion_AVEdataset.py --dataset /path/to/dataset_root --batch-size 32 --epochs 50
```
The audio-only, EMG-only, and lip-only baselines reside in `CSR_audio_only`, `CSR_emg_only`, and `CSR_lip_only`, respectively.

Pre-trained checkpoints for both tasks are available in `pretrained-models/` and can be specified via the corresponding `--path` arguments.

## Citation
If you use the source code in your work, please cite it as:
```bibtex
@article{zhou2025ave,
  title={AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals},
  author={Zhou, Dongliang and Zhang, Yakun and Wu, Jinghan and Zhang, Xingyu and Xie, Liang and Yin, Erwei},
  journal={IEEE Transactions on Human-Machine Systems},
  year={2025}
}
```

