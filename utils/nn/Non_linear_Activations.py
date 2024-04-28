import torch

from torch import nn


class AAM_Softmax(nn.Module):
    def __init__(self, num_speaker, model_ouput=192, prescale=30, margin=0.2):
        super().__init__()
        self.num_speaker = num_speaker
        self.Linear = nn.Linear(model_ouput, num_speaker, bias=False)
        self.scale = prescale
        self.margin = margin
        self.matrix = torch.eye(self.num_speaker).to(dtype=torch.bool).unsqueeze(0)

    def forward(self, x):
        matmul = self.Linear(x)
        x = torch.norm(x, dim=1)
        weight = torch.norm(self.Linear.weight, dim=1)
        cos_theta = matmul / (x.unsqueeze(-1) * weight.unsqueeze(0))
        cos_theta = cos_theta.unsqueeze(1)
        scale_cos_theta = self.scale * cos_theta
        theta = torch.acos(cos_theta)
        index = self.scale * torch.cos(theta + self.margin)
        exp_index = torch.where(self.matrix, index, scale_cos_theta)
        exp_matrix = torch.exp(exp_index)
        denominator = torch.sum(exp_matrix, dim=2, keepdim=False)
        aam_softmax = torch.exp(index).squeeze(1) / denominator
        return torch.log(aam_softmax)
