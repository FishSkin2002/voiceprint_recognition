import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.functional as F


class VoxCeleb1_set(Dataset):
    def __init__(self, data_path='./dataset/VoxCeleb1', subset='train', max_duration=2, target_sample_rate=16000, use_mono=True):
        super().__init__()
        self.subset = subset
        self.wave_dir = os.path.join(data_path, './wav')
        self.metadata = os.path.join(data_path, './iden_split.txt')
        self.waveforms_list = []
        self.id_dict = {}
        self.target_sample_rate = target_sample_rate
        self.use_mono = use_mono
        self.max_duration = max_duration
        if subset == 'train':
            self.subsetIndex = '1'
        elif subset == 'dev':
            self.subsetIndex = '2'
        elif subset == 'test':
            self.subsetIndex = '3'
        else:
            raise ValueError("子集仅能选择train、dev和test！")
        print(f'正在加载{self.subset}集')
        with open(self.metadata, 'r') as files:
            for file in files:
                subset, path = file.split()
                if subset == self.subsetIndex:
                    id = path.split('/')[0]
                    path = os.path.join(self.wave_dir, path)
                    self.waveforms_list.append((id, path))
                    if id not in self.id_dict:
                        self.id_dict[id] = len(self.id_dict)
        print(f'{self.subset}集加载完毕')

    def __getitem__(self, index):
        id, path = self.waveforms_list[index]
        waveform, sample_rate = torchaudio.load(uri=path, backend='soundfile')
        if sample_rate != self.target_sample_rate:
            waveform = F.resample(waveform, sample_rate, self.target_sample_rate)
        waveform_length = self.max_duration * self.target_sample_rate
        if waveform.shape[1] > waveform_length:
            random_number = random.randint(0, waveform.shape[1] - waveform_length)
            waveform = waveform[:, random_number:random_number + waveform_length]
        if self.use_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=False)
        label = self.id_dict[id]
        return waveform, label

    def __len__(self):
        return len(self.waveforms_list)

    def get_num_speaker(self):
        return len(self.id_dict)
