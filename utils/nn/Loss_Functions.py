import torch
from torch import nn
import torch.nn.functional as F


class AAMLoss(nn.Module):
    def __init__(self, scale=30, margin=0.2):
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, cos_theta, label):
        theta = torch.acos(cos_theta)
        label_index = self.scale * torch.cos(theta + self.margin)
        other_index = self.scale * cos_theta
        one_hot = F.one_hot(label, self.num_speaker).to(torch.bool)
        logits = torch.where(one_hot, label_index, other_index)
        return F.cross_entropy(logits, label)
