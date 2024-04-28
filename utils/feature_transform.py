import torch
from torch import nn
import torchaudio.transforms as T
from torchaudio.compliance.kaldi import fbank


class feature_transform(nn.Module):
    def __init__(self,
                 use_feature_extraction='MFCC',
                 feature_extraction_cfg=None,
                 spec_augument_cfg=None):
        super().__init__()
        self.use_feature_extraction = use_feature_extraction
        self.feature_extraction_cfg = feature_extraction_cfg
        self.spec_augument_cfg = spec_augument_cfg

        if self.use_feature_extraction == 'Spectrogram':
            self.extractor = T.Spectrogram(**self.feature_extraction_cfg['Spectrogram'])
        elif self.use_feature_extraction == 'MFCC':
            self.extractor = T.MFCC(**self.feature_extraction_cfg['MFCC'])
        elif self.use_feature_extraction == 'fbank':
            pass
        else:
            raise ValueError(f'不支持{self.use_feature_extraction}特征提取方式')

        if 'time_masking' in self.spec_augument_cfg:
            self.time_masking = T.TimeMasking(**self.spec_augument_cfg['time_masking'])
        if 'freq_masking' in self.spec_augument_cfg:
            self.freq_masking = T.FrequencyMasking(**self.spec_augument_cfg['freq_masking'])

    def forward(self, waveform):
        with torch.no_grad():
            feature = self.extractor(waveform)
            if self.spec_augument_cfg:
                feature = self.__spec_augument(feature)
        return feature

    def __spec_augument(self, feature):
        with torch.no_grad():
            if self.spec_augument_cfg.get('use_mfcc_cms') and self.use_feature_extraction == 'MFCC':
                mfcc_mean = torch.mean(feature, dim=-1, keepdim=True)
                feature = feature - mfcc_mean
            if 'time_masking' in self.spec_augument_cfg:
                feature = self.time_masking(feature)
            if 'frequency_masking' in self.spec_augument_cfg:
                feature = self.frequency_masking(feature)
        return feature
