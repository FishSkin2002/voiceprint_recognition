import torch

from torch import nn


class AAM_Softmax(nn.Module):
    def __init__(self, num_speaker, model_ouput=192, scale=30, margin=0.2, device=torch.device('cpu')):
        super().__init__()
        self.num_speaker = num_speaker
        self.Linear = nn.Linear(model_ouput, num_speaker, bias=False)
        self.scale = scale
        self.margin = margin
        self.device = device
        self.matrix = torch.eye(self.num_speaker, device=self.device).unsqueeze(0)

    def forward(self, x):
        matmul = self.Linear(x)
        x = torch.norm(x, dim=1)
        weight = torch.norm(self.Linear.weight, dim=1)
        cos_theta = matmul / (x.unsqueeze(-1) * weight.unsqueeze(0))
        cos_theta = cos_theta.unsqueeze(1)
        theta = torch.acos(cos_theta)
        index = self.scale * torch.cos(theta + self.margin)
        exp_index = self.matrix * index + (1 - self.matrix) * self.scale * cos_theta
        exp_matrix = torch.exp(exp_index)
        denominator = torch.sum(exp_matrix, dim=2, keepdim=False)
        aam_softmax = torch.exp(index).squeeze(1) / denominator
        return aam_softmax
