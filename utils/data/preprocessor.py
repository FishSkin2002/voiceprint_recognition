import math
import random

import torch
import torchaudio.functional as F
import torchaudio.transforms as T
from torch import nn


class preprocessor(nn.Module):
    def __init__(self,
                 subset,
                 max_duration,
                 target_sample_rate,
                 keep_audio_channel,
                 use_speed_perturbation,
                 add_noise,
                 speed_perturbation_sequence,
                 max_snr,
                 min_snr):
        super().__init__()

        self.subset = subset
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        self.keep_audio_channel = keep_audio_channel
        self.use_speed_perturbation = use_speed_perturbation
        self.add_noise = add_noise
        self.waveform_length = math.floor(self.max_duration * self.target_sample_rate)

        if self.use_speed_perturbation:
            speed_perturbation_sequence = list(speed_perturbation_sequence)
            self.speed_perturbation = T.SpeedPerturbation(self.target_sample_rate, speed_perturbation_sequence)

        if self.add_noise:
            self.max_snr = max_snr
            self.min_snr = min_snr

    def forward(self, waveform, sample_rate):
        with torch.no_grad():
            waveform = self.__audio_augument(waveform)

            if sample_rate != self.target_sample_rate:
                waveform = F.resample(waveform, sample_rate, self.target_sample_rate)

            waveform = self.__audio_crop(waveform)
        return waveform

    def __audio_crop(self, waveform):
        with torch.no_grad():

            if waveform.shape[1] > self.waveform_length:
                random_number = random.randint(0, waveform.shape[1] - self.waveform_length)
                waveform = waveform[:, random_number:random_number + self.waveform_length]
            elif waveform.shape[1] < self.waveform_length:
                zeros_tensor = torch.zeros([waveform.shape[0], self.waveform_length])
                zeros_tensor[0:waveform.shape[1]] = waveform
                waveform = zeros_tensor

            if not self.keep_audio_channel:
                waveform = torch.mean(waveform, dim=0, keepdim=False)

        return waveform

    def __audio_augument(self, waveform):
        with torch.no_grad():

            if self.use_speed_perturbation:
                waveform, _ = self.speed_perturbation(waveform)

            if self.add_noise:
                noise = torch.randn_like(waveform) + torch.rand_like(waveform)
                snr = torch.randint(low=self.min_snr, high=self.max_snr, size=[1])
                waveform = F.add_noise(waveform, noise, snr)
        return waveform
