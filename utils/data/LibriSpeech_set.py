import os

import torch
import torchaudio
from torch.utils.data import Dataset

from utils.data.preprocessor import preprocessor


class LibriSpeech_set(Dataset):
    def __init__(self,
                 subset,
                 preprocess_cfg,
                 data_path='./dataset/LibriSpeech'):
        super().__init__()
        self.subset = subset
        self.preprocessor = preprocessor(self.subset, **preprocess_cfg)
        self.waveforms_list = []
        self.idx_dict = {}
        self.subset_dir = os.path.join(data_path, subset)
        self.traverse_folder(self.subset_dir)

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

    def traverse_folder(self, root_folder):
        for file_name in os.listdir(root_folder):
            file_path = os.path.join(root_folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.flac'):
                idx = file_name.split('-')[0]
                self.waveforms_list.append((idx, file_path))
                if idx not in self.idx_dict:
                    self.idx_dict[idx] = len(self.idx_dict)
            elif os.path.isdir(file_path):
                self.traverse_folder(file_path)
