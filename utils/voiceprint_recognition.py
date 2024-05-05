import os
import shutil
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from utils.data.VoxCeleb1_set import VoxCeleb1_set
from utils.download.ERes2Net import ERes2Net
from utils.metrics import accuracy, cal_fpr_fnr, cal_eer_threshold, cal_min_dcf_threshold
from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.models.classifier import cosine_classifier
from utils.nn.Loss_Functions import AAMLoss, AMLoss
from utils.transform.feature_transform import feature_transform


class voiceprint_recognition:
    def __init__(self,
                 use_gpu,
                 use_feature_extraction,
                 use_model,
                 use_loss,
                 use_optimizer,
                 use_lr_scheduler,
                 train_epoch,
                 dataset_cfg,
                 dataloader_cfg,
                 feature_extraction_cfg,
                 model_cfg,
                 scheduler_cfg,
                 logdir='./logs',
                 save_model_path='./trained_models'):
        if use_gpu:
            assert torch.cuda.is_available(), 'CUDA不可用'
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.use_feature_extraction = use_feature_extraction
        self.use_model = use_model
        self.use_loss = use_loss
        self.use_optimizer = use_optimizer
        self.use_lr_scheduler = use_lr_scheduler
        self.train_epoch = train_epoch
        self.save_model_path = save_model_path
        self.writer = SummaryWriter(os.path.join(logdir, use_model))
        self.train_set = None
        self.dev_set = None
        self.num_speaker = None
        self.train_loader = None
        self.dev_loader = None
        self.transform = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.total_train_step = 0
        self.total_dev_step = 0
        self.best_eer = 1
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.feature_extraction_cfg = feature_extraction_cfg
        self.model_cfg = model_cfg
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
            acc, eer, threshold, min_dcf, min_dcf_threshold = self.evaluate()
            self.__save_model(i + 1, acc, eer, threshold, min_dcf, min_dcf_threshold)
        self.writer.close()

    def __set_data(self):
        self.train_set = VoxCeleb1_set('train', **self.dataset_cfg)
        self.dev_set = VoxCeleb1_set('dev', **self.dataset_cfg)
        print(f'数据集加载完成，训练集{len(self.train_set)}语音片段，'
              f'验证集{len(self.dev_set)}语音片段')
        self.num_speaker = self.train_set.get_num_speaker()
        self.train_loader = DataLoader(self.train_set, shuffle=True, **self.dataloader_cfg)
        self.dev_loader = DataLoader(self.dev_set, shuffle=False, **self.dataloader_cfg)

    def __set_feature_extractor(self):
        self.transform = feature_transform(self.use_feature_extraction, **self.feature_extraction_cfg)
        self.transform = self.transform.to(device=self.device)

    def __set_model(self):
        if self.use_model == 'ECAPA_TDNN':
            self.model = nn.Sequential(OrderedDict([
                ('backbone', ECAPA_TDNN(**self.model_cfg.get('ECAPA_TDNN', {}))),
                ('classifier', cosine_classifier(self.num_speaker))
            ]))
            self.model = self.model.to(device=self.device)
            summary(self.model, input_size=[2, 80, 201])
        elif self.use_model == 'ERes2Net':
            self.model = nn.Sequential(OrderedDict([
                ('backbone', ERes2Net(**self.model_cfg.get('ERes2Net', {}))),
                ('classifier', cosine_classifier(self.num_speaker))
            ]))
            self.model = self.model.to(device=self.device)
            summary(self.model, input_size=[2, 298, 80])
        else:
            raise ValueError(f'不支持{self.use_model}模型')

    def __set_criterion(self):
        if self.use_loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.use_loss == 'NLLoss':
            self.criterion = nn.NLLLoss()
        elif self.use_loss == 'AAMLoss':
            self.criterion = AAMLoss(self.num_speaker, **self.model_cfg.get('AAMLoss', {}))
        elif self.use_loss == 'AMLoss':
            self.criterion = AMLoss(self.num_speaker, **self.model_cfg.get('AMLoss', {}))
        else:
            raise ValueError(f'不支持{self.use_loss}损失函数')
        self.criterion = self.criterion.to(device=self.device)

    def __set_optimizer(self):
        if self.use_optimizer == 'Adam':
            self.optimizer = optim.Adam([
                {'params': self.model.backbone.parameters(), 'weight_decay': 2e-5},
                {'params': self.model.classifier.parameters(), 'weight_decay': 2e-4}
            ])
        elif self.use_optimizer == 'SGD':
            self.optimizer = optim.SGD(params=self.model.parameters(), momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f'不支持{self.use_optimizer}优化器')

        if self.use_lr_scheduler == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                         **self.scheduler_cfg.get('CyclicLR', {}))
        if self.use_lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  **self.scheduler_cfg.get('CosineAnnealingLR', {}))
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
            labels = labels.to(device=self.device)
            loss = self.criterion(outputs, labels)
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

    def __save_model(self, now_epoch, acc, eer, eer_threshold, min_dcf, min_dcf_threshold):
        model_path = os.path.join(self.save_model_path, self.use_model, f'epoch_{now_epoch}')
        if eer < self.best_eer:
            self.best_eer = eer
            model_path = os.path.join(self.save_model_path, f'{self.use_model}/best_model')
        too_old_model_path = os.path.join(self.save_model_path, self.use_model, f'epoch_{now_epoch - 3}')
        if os.path.exists(too_old_model_path):
            shutil.rmtree(too_old_model_path)
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.backbone.state_dict(), os.path.join(model_path, f'{self.use_model}.pth'))
        with open(os.path.join(self.save_model_path, f'{self.use_model}/metrics.txt'), 'a') as file:
            if eer == self.best_eer:
                file.write(f'Epoch: {now_epoch},    '
                           f'ACC: {acc:.10f},    '
                           f'EER: {eer:.10f},    '
                           f'EER threshold: {eer_threshold:.10f}    '
                           f'minDCF: {min_dcf:.10f},    '
                           f'minDCF threshold: {min_dcf_threshold:.10f}    '
                           f'best_model\n')
            else:
                file.write(f'Epoch: {now_epoch},    '
                           f'ACC: {acc:.10f},    '
                           f'EER: {eer:.10f},    '
                           f'EER threshold: {eer_threshold:.10f}    '
                           f'minDCF: {min_dcf:.10f},    '
                           f'minDCF threshold: {min_dcf_threshold:.10f}\n')
        print(f'模型成功保存到{model_path}')

    def evaluate(self):
        print('验证阶段开始')
        self.model.eval()
        all_outputs = torch.empty(0)
        all_labels = torch.empty(0)
        all_identities = torch.empty(0)
        with torch.no_grad():
            print('正在处理验证集……')
            for waveforms, labels in self.dev_loader:
                waveforms = waveforms.to(device=self.device)
                labels = labels.to(device=self.device)
                features = self.transform(waveforms)
                identities = self.model.backbone(features)
                outputs = self.model.classifier(identities)
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_identities = torch.cat([all_identities, identities], dim=0)
                all_outputs = torch.cat([all_outputs, outputs], dim=0)
            print('验证集处理完毕')
            print('正在计算模型指标……')
            acc = accuracy(all_outputs, all_labels)
            all_identities = F.normalize(all_identities, dim=-1)
            score_matrix = F.linear(all_identities, all_identities)
            labels_matrix = torch.eq(all_labels.unsqueeze(0), all_labels.unsqueeze(1))
            all_score = score_matrix.view(-1)
            all_labels = labels_matrix.view(-1)
            fpr, fnr, all_score = cal_fpr_fnr(all_score, all_labels)
            eer, eer_threshold = cal_eer_threshold(fpr, fnr, all_score)
            min_dcf, min_dcf_threshold = cal_min_dcf_threshold(fpr, fnr, all_score)
            print(f'Speaker Identification    '
                  f'ACC: {acc:.4%}\n'
                  f'Speaker Verification    '
                  f'EER: {eer:.4%}    '
                  f'EER threshold: {eer_threshold:.8f}    '
                  f'minDCF: {min_dcf:.8f}    '
                  f'minDCF threshold: {min_dcf_threshold:.8f}')
            self.total_dev_step += 1
            self.writer.add_scalar(tag='dev/ACC',
                                   scalar_value=acc,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/EER',
                                   scalar_value=eer,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/EER_threshold',
                                   scalar_value=eer_threshold,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/minDCF',
                                   scalar_value=min_dcf,
                                   global_step=self.total_dev_step)
            self.writer.add_scalar(tag='dev/minDCF_threshold',
                                   scalar_value=min_dcf_threshold,
                                   global_step=self.total_dev_step)
            return acc, eer, eer_threshold, min_dcf, min_dcf_threshold
