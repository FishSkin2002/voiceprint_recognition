import torch
import torchaudio.transforms as T
from torch import nn
from torchaudio.compliance.kaldi import fbank

from utils.transform.Augmentations import Augmentations


class feature_transform(nn.Module):
    def __init__(self,
                 use_feature_extraction,
                 use_mfcc_cms,
                 feature_extraction_cfg,
                 feature_transpose=False,
                 Augmentations_cfg=None):
        super().__init__()
        self.use_feature_extraction = use_feature_extraction
        self.feature_extraction_cfg = feature_extraction_cfg
        self.use_mfcc_cms = use_mfcc_cms
        self.Augmentations_cfg = Augmentations_cfg
        self.feature_transpose = feature_transpose

        if self.use_feature_extraction == 'Spectrogram':
            self.extractor = T.Spectrogram(**self.feature_extraction_cfg['Spectrogram'])
        elif self.use_feature_extraction == 'MFCC':
            self.extractor = T.MFCC(**self.feature_extraction_cfg['MFCC'])
        elif self.use_feature_extraction == 'fbank':
            pass
        else:
            raise ValueError(f'不支持{self.use_feature_extraction}特征提取方式')

        if self.Augmentations_cfg:
            self.aug = Augmentations(**self.Augmentations_cfg)

    def forward(self, waveforms):
        with torch.no_grad():
            if self.use_feature_extraction == 'fbank':
                features = self.__cal_fbank(waveforms)
            else:
                features = self.extractor(waveforms)

            if self.use_mfcc_cms and self.use_feature_extraction == 'MFCC':
                mfcc_mean = torch.mean(features, dim=-1, keepdim=True)
                features = features - mfcc_mean

            if self.Augmentations_cfg:
                self.aug(features)

            if self.feature_transpose:
                features = torch.transpose(features, 1, 2)

        return features

    def __cal_fbank(self, waveforms):
        features = torch.empty(0)
        for i in range(waveforms.shape[0]):
            feature = fbank(waveforms[i], **self.feature_extraction_cfg['fbank']).unsqueeze(0)
            features = torch.cat([features, feature], dim=0)
        return features
