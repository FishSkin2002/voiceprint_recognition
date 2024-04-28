import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.functional as F
import torchaudio.transforms as T


class VoxCeleb1_set(Dataset):
    def __init__(self,
                 subset,
                 data_path='./dataset/VoxCeleb1',
                 max_duration=2,
                 target_sample_rate=16000,
                 use_mono=True,
                 stack_utterances=False,
                 stack_num=20,
                 audio_augument_cfg=None):
        super().__init__()
        self.subset = subset
        self.target_sample_rate = target_sample_rate
        self.use_mono = use_mono
        self.max_duration = max_duration
        self.audio_augument_cfg = audio_augument_cfg
        self.wave_dir = os.path.join(data_path, './wav')
        self.metadata = os.path.join(data_path, './iden_split.txt')
        self.waveforms_list = []
        self.idx_dict = {}
        self.stack_utterances = stack_utterances
        self.stack_num = stack_num
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

        if self.audio_augument_cfg.get('use_speed_perturbation'):
            speed_perturbation_sequence = list(self.audio_augument_cfg['speed_perturbation_sequence'])
            self.speed_perturbation = T.SpeedPerturbation(self.target_sample_rate, speed_perturbation_sequence)
        if self.audio_augument_cfg.get('add_noise'):
            self.snr_range = self.audio_augument_cfg.get('snr_range')

    def __getitem__(self, index):
        with torch.no_grad():
            idx, path = self.waveforms_list[index]
            waveform, sample_rate = torchaudio.load(uri=path, backend='soundfile')
            if self.subset == 'train' and self.audio_augument_cfg:
                waveform = self.__audio_augument(waveform)
            if sample_rate != self.target_sample_rate:
                waveform = F.resample(waveform, sample_rate, self.target_sample_rate)
            waveform = self.__audio_crop(waveform)
            label = self.idx_dict[idx]
        return waveform, label

    def __len__(self):
        return len(self.waveforms_list)

    def get_num_speaker(self):
        return len(self.idx_dict)

    def __audio_crop(self, waveform):
        with torch.no_grad():
            waveform_length = self.max_duration * self.target_sample_rate
            if waveform.shape[1] > waveform_length:
                random_number = random.randint(0, waveform.shape[1] - waveform_length)
                waveform = waveform[:, random_number:random_number + waveform_length]
            elif waveform.shape[1] < waveform_length:
                zeros_tensor = torch.zeros([waveform.shape[0], waveform_length])
                zeros_tensor[0:waveform.shape[1]] = waveform
                waveform = zeros_tensor
            if self.stack_utterances:
                waveform = waveform.unsqueeze(1)
                waveform = torch.cat([waveform] * self.stack_num, dim=1)
            if self.use_mono:
                waveform = torch.mean(waveform, dim=0, keepdim=False)
        return waveform

    def __audio_augument(self, waveform):
        with torch.no_grad():
            if self.audio_augument_cfg.get('use_speed_perturbation'):
                waveform, _ = self.speed_perturbation(waveform)
            if self.audio_augument_cfg.get('add_noise'):
                noise = torch.randn_like(waveform) + torch.rand_like(waveform)
                snr = torch.randint(low=self.snr_range[0], high=self.snr_range[1], size=[1])
                waveform = F.add_noise(waveform, noise, snr)
        return waveform
