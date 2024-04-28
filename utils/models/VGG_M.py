import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class VGG_M(nn.Module):
    def __init__(self, num_speaker=1251):
        super().__init__()
        self.num_speaker = num_speaker
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2)),
            ('mpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1)),
            ('mpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('mpool5', nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))),
            ('flatten', nn.Flatten()),
            ('fc6', nn.Linear(in_features=256 * 9 * 8, out_features=4096 * 8)),
            ('apool6', nn.AvgPool1d(kernel_size=8)),
            ('fc7', nn.Linear(in_features=4096, out_features=1024))
        ]))

    def forward(self, x):
        x = self.model(x)
        return x
