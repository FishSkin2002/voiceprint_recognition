from torch import nn
import torch.nn.functional as F


class cosine_classifier(nn.Module):
    def __init__(self, num_speaker, embedding_dim=192):
        super().__init__()
        self.num_speaker = num_speaker
        self.Linear = nn.Linear(embedding_dim, num_speaker, bias=False)
        self.W = self.Linear.weight

    def forward(self, x):
        cos_theta = F.linear(F.normalize(x, dim=-1), F.normalize(self.W, dim=-1))
        return cos_theta
