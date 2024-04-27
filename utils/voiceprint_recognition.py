import os

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.feature_transform import feature_transform
from utils.nn.Non_linear_Activations import AAM_Softmax
from utils.data.VoxCeleb1_set import VoxCeleb1_set
from utils.metrics import accuracy, fpr_fnr, eer_threshold
from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.models.VGG_M import VGG_M


class voiceprint_recognition:
    def __init__(self,
                 use_gpu=True,
                 use_feature_extraction='MFCC',
                 use_model='ECAPA_TDNN',
                 use_loss='AAMLoss',
                 # use_veri_loss='MSELoss',
                 # use_iden_loss='NLLLoss',
                 # veri_iden_loss_weight=(1, 1),
                 use_optimizer='Adam',
                 use_lr_scheduler='CyclicLR',
                 target_sample_rate=16000,
                 train_epoch=60,
                 logdir='./logs',
                 save_model_path='./trained_models',
                 dataset_cfg=None,
                 dataloader_cfg=None,
                 transform_cfg=None,
                 scheduler_cfg=None):
        if use_gpu:
            assert torch.cuda.is_available(), 'CUDA不可用'
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.use_feature_extraction = use_feature_extraction
        self.use_model = use_model
        self.use_loss = use_loss
        # self.use_veri_loss = use_veri_loss
        # self.use_iden_loss = use_iden_loss
        # self.veri_iden_loss_weight = veri_iden_loss_weight
        self.use_optimizer = use_optimizer
        self.use_lr_scheduler = use_lr_scheduler
        self.target_sample_rate = target_sample_rate
        self.train_epoch = train_epoch
        self.save_model_path = save_model_path
        self.writer = SummaryWriter(os.path.join(logdir, use_model))
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.num_speaker = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.transform = None
        self.model = None
        self.criterion_veri = None
        self.criterion_iden = None
        self.optimizer = None
        self.scheduler = None
        self.total_train_step = 0
        self.total_dev_step = 0
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.transform_cfg = transform_cfg
        self.scheduler_cfg = scheduler_cfg

    def train(self):
        self.__set_data()
        self.__set_feature_extractor()
        self.__set_model()
        self.__set_criterion()
        self.__set_optimizer()
        for i in range(self.train_epoch):
            self.model.train()
            self.__epoch(i + 1)
            acc, eer, threshold = self.evaluate()
            self.__save_model(i + 1, acc, eer, threshold)
        self.writer.close()

    def __set_data(self):
        self.train_set = VoxCeleb1_set('train', **self.dataset_cfg)
        self.dev_set = VoxCeleb1_set('dev', **self.dataset_cfg)
        self.test_set = VoxCeleb1_set('test', **self.dataset_cfg)
        print(
            f'数据集加载完成，训练集{len(self.train_set)}语音片段，验证集{len(self.dev_set)}语音片段，测试集{len(self.test_set)}语音片段')
        self.num_speaker = self.train_set.get_num_speaker()
        self.train_loader = DataLoader(self.train_set, **self.dataloader_cfg)
        self.dev_loader = DataLoader(self.dev_set, **self.dataloader_cfg)
        self.test_loader = DataLoader(self.test_set, **self.dataloader_cfg)

    def __set_feature_extractor(self):
        self.transform = feature_transform(**self.transform_cfg)
        self.transform = self.transform.to(device=self.device)

    def __set_model(self):
        if self.use_model == 'ECAPA_TDNN':
            backbone = ECAPA_TDNN()
            softmax = AAM_Softmax(self.num_speaker)
            self.model = nn.Sequential(OrderedDict([
                ('backbone', backbone),
                ('softmax', softmax)
            ]))
            summary(self.model, input_size=[1, 80, 201])
        elif self.use_model == 'VGG_M':
            self.model = VGG_M()
        else:
            raise ValueError(f'不支持{self.use_model}模型')
        self.model = self.model.to(device=self.device)

    def __set_criterion(self):
        if self.use_loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.use_loss == 'NLLoss':
            self.criterion = nn.NLLLoss()
        else:
            raise ValueError(f'不支持{self.use_loss}损失函数')
        self.criterion = self.criterion.to(device=self.device)
        # if self.use_veri_loss == 'MSELoss':
        #     self.criterion_veri = nn.MSELoss()
        # elif self.use_veri_loss == 'CosineEmbeddingLoss':
        #     self.criterion_veri = nn.CosineEmbeddingLoss()
        # else:
        #     raise ValueError(f'不支持{self.use_veri_loss}损失函数')
        #
        # if self.use_iden_loss == 'NLLLoss':
        #     self.criterion_iden = nn.NLLLoss()
        # elif self.use_iden_loss == 'CrossEntropyLoss':
        #     self.criterion_iden = nn.CrossEntropyLoss()
        # else:
        #     raise ValueError(f'不支持{self.use_iden_loss}损失函数')
        #
        # self.criterion_veri = self.criterion_veri.to(device=self.device)
        # self.criterion_iden = self.criterion_iden.to(device=self.device)

    def __set_optimizer(self):
        if self.use_optimizer == 'Adam':
            self.optimizer = optim.Adam(
                [{'params': self.model.backbone.parameters(), 'weight_decay': 2e-5},
                 {'params': self.model.softmax.parameters(), 'weight_decay': 2e-4}]
            )
        else:
            raise ValueError(f'不支持{self.use_optimizer}优化器')

        if self.use_lr_scheduler == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, **self.scheduler_cfg)
        else:
            raise ValueError(f'不支持{self.use_lr_scheduler}学习率衰减方法')

    def __epoch(self, now_epoch):
        print(f'开始Epoch[{now_epoch}/{self.train_epoch}]的训练……')
        all_loss = []
        for batch_id, (waveforms, labels) in enumerate(self.train_loader):
            self.total_train_step += 1
            self.optimizer.zero_grad()
            waveforms = waveforms.to(device=self.device)
            labels = labels.to(device=self.device)
            features = self.transform(waveforms)
            outputs = self.model(features)
            loss = self.criterion(torch.log(outputs), labels)
            # identities = self.model.backbone(features)
            # outputs = self.model.softmax(identities)
            # loss_iden = self.criterion_iden(torch.log(outputs), labels) * torch.tensor(self.veri_iden_loss_weight[1])
            # cosine_matrix = F.cosine_similarity(identities.unsqueeze(0), identities.unsqueeze(1), dim=-1)
            # labels_matrix = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
            # ignore_matrix = torch.min(cosine_matrix, torch.zeros_like(cosine_matrix) + 0.5)
            # labels_matrix = torch.where(labels_matrix, torch.ones_like(labels_matrix), ignore_matrix)
            # loss_veri = self.criterion_veri(cosine_matrix, labels_matrix) * torch.tensor(self.veri_iden_loss_weight[0])
            # loss = (loss_veri + loss_iden) / torch.sum(torch.tensor(self.veri_iden_loss_weight))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            all_loss.append(loss)
            if (batch_id + 1) % 100 == 0:
                print(
                    f'Epoch:[{now_epoch}/{self.train_epoch}], '
                    f'Batch:[{batch_id + 1}/{len(self.train_loader)}], '
                    f'Loss:{sum(all_loss) / len(all_loss):.4f}, '
                    f'Learning Rate: {self.scheduler.get_last_lr()[0]:.10f}'
                )
                all_loss = []
            self.writer.add_scalar(tag='train/loss',
                                   scalar_value=loss,
                                   global_step=self.total_train_step)
            self.writer.add_scalar(tag='train/learning_rate',
                                   scalar_value=self.scheduler.get_last_lr()[0],
                                   global_step=self.total_train_step)

    def __save_model(self, now_epoch, acc, eer, threshold):
        model_path = os.path.join(self.save_model_path, f'{self.use_model}/epoch_{now_epoch}')
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, f'{self.use_model}.pth'))
        with open(os.path.join(self.save_model_path, f'{self.use_model}/metrics.txt'), 'a') as file:
            file.write(f'Epoch: {now_epoch},    '
                       f'ACC: {acc:.10f},    '
                       f'EER: {eer:.10f},    '
                       f'threshold: {threshold:.10f}\n')
        print(f'模型成功保存到{model_path}')

    def evaluate(self):
        print('验证阶段开始')
        self.model.eval()
        acc = []
        all_score = None
        all_labels = None
        with torch.no_grad():
            print('正在处理验证集……')
            for waveforms, labels in self.dev_loader:
                waveforms = waveforms.to(device=self.device)
                labels = labels.to(device=self.device)
                features = self.transform(waveforms)
                identities = self.model.backbone(features)
                outputs = self.model.softmax(identities)
                acc.append(accuracy(outputs, labels))
                score_matrix = F.cosine_similarity(identities.unsqueeze(0), identities.unsqueeze(1), dim=-1)
                labels_matrix = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
                all_score = score_matrix.view(-1)
                all_labels = labels_matrix.view(-1)
                if all_score is None:
                    all_score = score_matrix.view(-1)
                else:
                    all_score = torch.cat([all_score, score_matrix.view(-1)], dim=0)
                if all_labels is None:
                    all_labels = labels_matrix.view(-1)
                else:
                    all_labels = torch.cat([all_labels, labels_matrix.view(-1)], dim=0)
            print('验证集处理完毕')
            print('正在计算模型指标……')
            acc = sum(acc) / len(acc)
            fpr, fnr, all_score = fpr_fnr(all_score, all_labels)
            eer, threshold = eer_threshold(fpr, fnr, all_score)
            print(f'Speaker Identification ACC: {acc:.4%}\n'
                  f'Speaker Verification EER: {eer:.4%}, '
                  f'threshold: {threshold:.8f}')
            self.total_dev_step += 1
            self.writer.add_scalar(tag='dev/identification/ACC',
                                   scalar_value=acc,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/verification/EER',
                                   scalar_value=eer,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/verification/threshold',
                                   scalar_value=threshold,
                                   global_step=self.total_dev_step)
            return acc, eer, threshold
