import torch.nn as nn
from collections import OrderedDict


class three_dim_CNN(nn.Module):
    def __init__(self, input_size, num_speaker=1251):
        super().__init__()
        self.input_size = input_size
        self.model = nn.Sequential(OrderedDict([
            ('Conv1-1', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 5), stride=(1, 1, 1))),
            ('Conv1-2', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 9, 1), stride=(1, 2, 1), padding=(1, 0, 0))),
            ('Pool1', nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))),
            ('Conv2-1', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 4), stride=(1, 1, 1))),
            ('Conv2-2', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 8, 1), stride=(1, 2, 1))),
            ('Pool2', nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))),
            ('Conv3-1', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 3), stride=(1, 1, 1))),
            ('Conv3-2', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 7, 1), stride=(1, 1, 1))),
            ('Conv4-1', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 1, 3), stride=(1, 1, 1))),
            ('Conv4-2', nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 7, 1), stride=(1, 1, 1))),
            ('Flatten', nn.Flatten()),
            ('FC5', nn.Linear(in_features=4 * 3 * 3 * 128, out_features=128))
        ]))

    def forward(self, x):
        x = self.model(x)
        return x
