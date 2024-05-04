import os

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.data.preprocessor import preprocessor


class VoxCeleb1_set(Dataset):
    def __init__(self,
                 subset,
                 preprocess_cfg,
                 data_path='./dataset/VoxCeleb1'):
        super().__init__()
        self.subset = subset
        self.preprocessor = preprocessor(self.subset, **preprocess_cfg)
        self.wave_dir = os.path.join(data_path, './wav')
        self.metadata = os.path.join(data_path, './iden_split.txt')
        self.waveforms_list = []
        self.idx_dict = {}

        if self.subset == 'train':
            self.subsetIndex = '1'
        elif self.subset == 'dev':
            self.subsetIndex = '2'
        elif self.subset == 'test':
            self.subsetIndex = '3'
        else:
            raise ValueError("子集仅能选择train、dev和test！")

        with open(self.metadata, 'r') as files:
            for file in files:
                subset, path = file.split()
                if subset == self.subsetIndex:
                    idx = path.split('/')[0]
                    path = os.path.join(self.wave_dir, path)
                    self.waveforms_list.append((idx, path))
                    if idx not in self.idx_dict:
                        self.idx_dict[idx] = len(self.idx_dict)

    def __getitem__(self, index):
        with torch.no_grad():
            idx, path = self.waveforms_list[index]
            waveform, sample_rate = torchaudio.load(uri=path)
            waveform = self.preprocessor(waveform, sample_rate)
            label = self.idx_dict[idx]
        return waveform, label

    def __len__(self):
        return len(self.waveforms_list)

    def get_num_speaker(self):
        return len(self.idx_dict)
