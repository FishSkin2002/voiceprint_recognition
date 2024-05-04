import torch
import torchaudio.transforms as T
from torch import nn


class Augmentations(nn.Module):
    def __init__(self,
                 time_masking,
                 time_mask_param,
                 freq_masking,
                 freq_mask_param,
                 iid_masks):
        super().__init__()
        self.time_masking = time_masking
        self.freq_masking = freq_masking
        if self.time_masking:
            self.time_masker = T.TimeMasking(time_mask_param, iid_masks)
        if self.time_masking:
            self.freq_masker = T.FrequencyMasking(freq_mask_param, iid_masks)

    def forward(self, feature):
        with torch.no_grad():

            if self.time_masking:
                feature = self.time_masker(feature)

            if self.freq_masking:
                feature = self.freq_masker(feature)

        return feature
