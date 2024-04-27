import torch

from torch import nn


class AAM_Softmax(nn.Module):
    def __init__(self, num_speaker, len_trainloader, device, model_ouput=192, prescale=30, margin=0.2):
        super().__init__()
        self.num_speaker = num_speaker
        self.Linear = nn.Linear(model_ouput, num_speaker, bias=False)
        self.scale = prescale
        self.margin = margin
        self.matrix = torch.eye(self.num_speaker, device=device).unsqueeze(0)
        self.count = 0
        self.update_iteration = 4 * len_trainloader

    def forward(self, x):
        self.count += 1
        if self.count == self.update_iteration:
            self.__scale_parameter()
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

    def __scale_parameter(self):
        self.scale = nn.Parameter(torch.tensor(self.scale, dtype=torch.float))
