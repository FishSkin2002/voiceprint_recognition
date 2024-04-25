import torch
import torchaudio
from torch import nn
import torch.nn.functional as F

from utils.feature_transform import feature_transform
from utils.metrics import accuracy, eer_threshold, fpr_fnr
from utils.nn.Non_linear_Activations import AAM_Softmax
from utils.nn.Pooling_layers import Attentive_Stat_Pooling

# test = torch.randn([8, 4])
# print(test)
# test = torch.chunk(test, 2, dim=0)
# print(test)
# all_score = torch.rand(10)
# all_label = torch.randint(low=0, high=5, size=[20])
# all_label = list(torch.chunk(all_label, 2))
# print(all_label)
# fpr, fnr, all_score = fpr_fnr(all_score, all_label)
# eer, threshold = eer_threshold(fpr, fnr, all_score)
# print(f'eer: {eer:.8f}, threshold: {threshold:.8f}')
# pool = Attentive_Stat_Pooling(1536, 128)
# test = torch.randn([2, 1536, 41])
# pool(test)
x = torch.randn([1, 192])
Linear = nn.Linear(192, 1251, bias=False)
with torch.no_grad():
    cos_theta = F.cosine_similarity(Linear.weight.unsqueeze(0), x.unsqueeze(1), dim=2)
    test = Linear(x) / (torch.norm(x) * torch.norm(Linear.weight, dim=1))
print(cos_theta)
print(test)
threshold = 1e-7
comparison = torch.abs(cos_theta - test) < threshold
print(comparison.all())