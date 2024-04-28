import torch
from torch import nn
import torch.nn.functional as F
from utils.nn.Pooling_layers import Attentive_Stat_Pooling
from collections import OrderedDict


class ECAPA_TDNN(nn.Module):
    def __init__(self, input_size=80, channels=512, bottleneck=128, embedding_dim=192):
        super().__init__()
        self.Conv1D_ReLU_BN = Conv1D_ReLU_BN(input_size, channels, kernel_size=5, padding=2, dilation=1)
        self.SE_Res2Block_List = nn.ModuleList([
            SE_Res2Block(channels, kernel_size=3, padding=2, dilation=2, bottleneck=bottleneck),
            SE_Res2Block(channels, kernel_size=3, padding=3, dilation=3, bottleneck=bottleneck),
            SE_Res2Block(channels, kernel_size=3, padding=4, dilation=4, bottleneck=bottleneck)
        ])
        self.cat_channels = channels * 3
        self.Conv1D_cat = nn.Conv1d(in_channels=self.cat_channels, out_channels=self.cat_channels, kernel_size=1,
                                    dilation=1)
        self.Attentive_Stat_Pooling_BN = Attentive_Stat_Pooling_BN(self.cat_channels, bottleneck)
        self.FC_BN = nn.Sequential(OrderedDict([
            ('FC', nn.Linear(in_features=self.cat_channels * 2, out_features=embedding_dim)),
            ('BN', nn.BatchNorm1d(num_features=embedding_dim))
        ]))

    def forward(self, x):
        x = self.Conv1D_ReLU_BN(x)
        out0 = self.SE_Res2Block_List[0](x)
        out1 = self.SE_Res2Block_List[1](out0)
        out2 = self.SE_Res2Block_List[2](out1)
        x = F.relu(self.Conv1D_cat(torch.cat((out0, out1, out2), dim=1)))
        x = self.Attentive_Stat_Pooling_BN(x)
        x = self.FC_BN(x)
        return x


class Conv1D_ReLU_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.Conv1D = nn.Conv1d(in_channels, out_channels, stride, kernel_size, padding, dilation)
        self.BatchNorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.Conv1D(x))
        x = self.BatchNorm(x)
        return x


class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, padding, dilation, bottleneck):
        super().__init__()
        self.Conv1D_ReLu_BN_List = nn.ModuleList([
            Conv1D_ReLU_BN(channels, channels, kernel_size=1),
            Conv1D_ReLU_BN(channels, channels, kernel_size=1)
        ])
        self.Res2_Dilated_Conv1D = Res2_Dilated_Conv1D(channels, kernel_size, padding, dilation, scale=8)
        self.BN = nn.BatchNorm1d(channels)
        self.SE_Block = SE_Block(channels, bottleneck)

    def forward(self, x):
        out = self.Conv1D_ReLu_BN_List[0](x)
        out = F.relu(self.Res2_Dilated_Conv1D(out))
        out = self.BN(out)
        out = self.Conv1D_ReLu_BN_List[1](out)
        out = self.SE_Block(out)
        return x + out


class Res2_Dilated_Conv1D(nn.Module):
    def __init__(self, channels, kernel_size, padding, dilation, scale=8):
        super().__init__()
        self.scale = scale
        self.Conv1d_1_List = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Conv1d(channels, channels, kernel_size=1)
        ])
        self.Conv1d_3 = nn.Conv1d(channels // scale, channels // scale, kernel_size, 1, padding, dilation)

    def forward(self, x):
        x = self.Conv1d_1_List[0](x)
        x = torch.chunk(x, chunks=self.scale, dim=1)
        y = []
        for i in range(self.scale):
            if i == 0:
                y.append(x[i])
            elif i == 1:
                y.append(self.Conv1d_3(x[i]))
            else:
                y.append(self.Conv1d_3(y[i - 1] + x[i]))
        y = torch.cat(y, dim=1)
        y = self.Conv1d_1_List[1](y)
        return y


class SE_Block(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.Global_pooling = nn.AdaptiveAvgPool1d(1)
        self.FC1 = nn.Linear(channels, bottleneck)
        self.FC2 = nn.Linear(bottleneck, channels)

    def forward(self, x):
        se = self.Global_pooling(x)
        se = se.squeeze(2)
        se = F.relu(self.FC1(se))
        se = F.sigmoid(self.FC2(se))
        se = se.unsqueeze(2)
        return x * se


class Attentive_Stat_Pooling_BN(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.Attentive_Stat_Pooling = Attentive_Stat_Pooling(channels, bottleneck)
        self.BatchNorm1d = nn.BatchNorm1d(channels * 2)

    def forward(self, x):
        x = self.Attentive_Stat_Pooling(x)
        x = self.BatchNorm1d(x)
        return x


class ECAPA_TDNN_classifier(nn.Module):
    def __init__(self, num_speaker, model_ouput=192):
        super().__init__()
        self.num_speaker = num_speaker
        self.Linear = nn.Linear(model_ouput, num_speaker, bias=False)
        self.W = self.Linear.weight

    def forward(self, x):
        cos_theta = F.linear(F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))
        return cos_theta
