import torch
import torch.nn as nn
import torch.nn.functional as F


class Attentive_Stat_Pooling(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.Linear1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.Linear2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x):
        alpha = F.relu(self.Linear1(x))
        alpha = F.softmax(self.Linear2(alpha), dim=2)
        mu = torch.sum(alpha * x, dim=2)
        sigma = torch.sqrt((torch.sum(alpha * x * x, dim=2) - mu * mu).clamp(min=1e-9))
        return torch.cat([mu, sigma], dim=1)
