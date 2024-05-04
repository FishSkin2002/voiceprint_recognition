import torch
import torchaudio.transforms as T
from torch import nn

from utils.transform.Augmentations import Augmentations


class feature_transform(nn.Module):
    def __init__(self,
                 use_feature_extraction,
                 use_mfcc_cms,
                 feature_extraction_cfg,
                 Augmentations_cfg):
        super().__init__()
        self.use_feature_extraction = use_feature_extraction
        self.feature_extraction_cfg = feature_extraction_cfg
        self.use_mfcc_cms = use_mfcc_cms
        self.Augmentations_cfg = Augmentations_cfg

        if self.use_feature_extraction == 'Spectrogram':
            self.extractor = T.Spectrogram(**self.feature_extraction_cfg['Spectrogram'])
        elif self.use_feature_extraction == 'MFCC':
            self.extractor = T.MFCC(**self.feature_extraction_cfg['MFCC'])
        else:
            raise ValueError(f'不支持{self.use_feature_extraction}特征提取方式')

        if self.Augmentations_cfg:
            self.aug = Augmentations(**self.Augmentations_cfg)

    def forward(self, waveform):
        with torch.no_grad():
            feature = self.extractor(waveform)

            if self.use_mfcc_cms and self.use_feature_extraction == 'MFCC':
                mfcc_mean = torch.mean(feature, dim=-1, keepdim=True)
                feature = feature - mfcc_mean

            if self.Augmentations_cfg:
                self.aug(feature)

        return feature
