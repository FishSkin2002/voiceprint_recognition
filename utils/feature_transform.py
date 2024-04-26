from typing import Optional

import torch
from torch import nn
import torchaudio.transforms as T
import torchaudio.functional as F


class feature_transform(nn.Module):
    def __init__(self,
                 add_noise=True,
                 snr_range=(40, 60),
                 use_feature_extraction='MFCC',
                 device=torch.device('cuda'),
                 sample_rate=16000,
                 use_mfcc_cms=False,
                 log_mel=False,
                 use_specaugument=False,
                 sakwargs: Optional[dict] = None):
        super().__init__()
        self.device = device
        self.add_noise = add_noise
        self.snr_range = snr_range
        self.use_feature_extraction = use_feature_extraction
        self.sample_rate = sample_rate
        self.log_mel = log_mel
        self.use_mfcc_cms = use_mfcc_cms
        self.use_specaugument = use_specaugument
        self.sakwargs = sakwargs
        if self.use_feature_extraction == 'Spectrogram':
            self.transform = T.Spectrogram(n_fft=512 * 2,
                                           win_length=self.sample_rate // 1000 * 25,
                                           hop_length=self.sample_rate // 1000 * 10,
                                           window_fn=torch.hamming_window,
                                           normalized=True)
        elif self.use_feature_extraction == 'MFCC':
            self.transform = T.MFCC(n_mfcc=80,
                                    log_mels=self.log_mel,
                                    melkwargs={'n_fft': self.sample_rate // 1000 * 25,
                                               'hop_length': self.sample_rate // 1000 * 10,
                                               'n_mels': 80
                                               })
        else:
            raise ValueError(f'不支持{self.use_feature_extraction}特征提取方式')
        self.transform.to(device=self.device)
        if use_specaugument:
            expected_sakwargs = {'time_masking', 'frequency_masking'}
            actual_keys = self.sakwargs.keys()
            unexpected_keys = [key for key in actual_keys if key not in expected_sakwargs]
            if unexpected_keys:
                raise ValueError("Unexpected SpecAugument keyword arguments:", unexpected_keys)
            if 'time_masking' in self.sakwargs:
                self.time_masking_transform = T.TimeMasking(self.sakwargs['time_masking'], True)
            if 'frequency_masking' in self.sakwargs:
                self.frequency_masking_transform = T.FrequencyMasking(self.sakwargs['frequency_masking'], True)

    def forward(self, x):
        with torch.no_grad():
            if self.add_noise:
                noise = torch.randn_like(x, device=self.device)
                snr = torch.randint(low=self.snr_range[0], high=self.snr_range[1], size=[1], device=self.device)
                x = F.add_noise(x, noise, snr)
            x = self.transform(x)
            if self.use_feature_extraction == 'MFCC':
                if self.use_mfcc_cms:
                    mfcc_mean = torch.mean(x, dim=-1, keepdim=True)
                    x = x - mfcc_mean
                if self.use_specaugument:
                    x = self.__spec_augument(x)
        return x

    def __spec_augument(self,
                        feature: torch.tensor):
        with torch.no_grad():
            if 'time_masking' in self.sakwargs:
                feature = self.time_masking_transform(feature)
            if 'frequency_masking' in self.sakwargs:
                feature = self.frequency_masking_transform(feature)
        return feature
