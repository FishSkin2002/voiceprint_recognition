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
                 use_model='ECAPA_TDNN',
                 use_feature_extraction='MFCC',
                 use_loss='NLLLoss',
                 use_optimizer='Adam',
                 use_lr_scheduler='CyclicLR',
                 data_path='./dataset/VoxCeleb1',
                 batch_size=128,
                 sample_rate=16000,
                 train_epoch=60,
                 log_dir='./logs',
                 save_model_path='./models'):
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise ValueError('GPU不可用')
        else:
            self.device = torch.device('cpu')
        print(f'当前使用设备：{self.device}')
        self.writer = SummaryWriter(os.path.join(log_dir, use_model))
        self.use_model = use_model
        self.use_feature_extraction = use_feature_extraction
        self.use_loss = use_loss
        self.use_optimizer = use_optimizer
        self.use_lr_scheduler = use_lr_scheduler
        self.data_path = data_path
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.num_speaker = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.feature_extractor = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_epoch = train_epoch
        self.total_train_step = 0
        self.total_dev_step = 0
        # self.criterion_iden = nn.NLLLoss()
        self.save_model_path = save_model_path

    def train(self):
        self.__set_data()
        self.__set_feature_extractor()
        self.__set_model()
        self.__set_criterion()
        self.__set_optimizer()
        for i in range(self.train_epoch):
            self.model.train()
            self.__epoch(i + 1)
            self.evaluate()
            self.__save_model(i + 1)
        self.writer.close()

    def __set_data(self):
        print('正在加载数据……')
        self.train_set = VoxCeleb1_set(self.data_path, 'train')
        self.dev_set = VoxCeleb1_set(self.data_path, 'dev')
        self.test_set = VoxCeleb1_set(self.data_path, 'test')
        self.num_speaker = self.train_set.get_num_speaker()
        self.train_loader = DataLoader(self.train_set,
                                       self.batch_size,
                                       shuffle=True,
                                       num_workers=16,
                                       pin_memory=True)
        self.dev_loader = DataLoader(self.dev_set,
                                     self.batch_size,
                                     shuffle=True,
                                     num_workers=16,
                                     pin_memory=True)
        self.test_loader = DataLoader(self.test_set,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=16,
                                      pin_memory=True)
        print('数据加载完成')

    def __set_feature_extractor(self):
        print('正在加载特征提取……')
        self.feature_extractor = feature_transform(self.use_feature_extraction,
                                                   self.device,
                                                   use_mfcc_cms=True,
                                                   log_mel=True,
                                                   use_specaugument=True,
                                                   sakwargs={'time_masking': 5,
                                                             'frequency_masking': 10})
        print(f'特征提取加载完成')

    def __set_model(self):
        print('正在加载模型……')
        if self.use_model == 'ECAPA_TDNN':
            backbone = ECAPA_TDNN()
            softmax = AAM_Softmax(self.num_speaker, device=self.device)
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
        print(f'模型{self.use_model}加载完成')

    def __set_criterion(self):
        print('正在加载损失函数……')
        if self.use_loss == 'NLLLoss':
            self.criterion = nn.NLLLoss()
        elif self.use_loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.use_loss == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif self.use_loss == 'CosineEmbeddingLoss':
            self.criterion = nn.CosineEmbeddingLoss(margin=0.5)
        else:
            raise ValueError(f'不支持{self.use_loss}损失函数')
        self.criterion = self.criterion.to(device=self.device)
        # self.criterion_iden = self.criterion_iden.to(device=self.device)
        print(f'损失函数{self.use_loss}加载完成')

    def __set_optimizer(self):
        print('正在加载优化器……')
        if self.use_optimizer == 'Adam':
            self.optimizer = optim.Adam(
                [{'params': self.model.backbone.parameters(), 'weight_decay': 2e-5},
                 {'params': self.model.softmax.parameters(), 'weight_decay': 2e-4}
                 ])
        else:
            raise ValueError(f'不支持{self.use_optimizer}优化器')
        print(f'优化器{self.use_optimizer}加载完成')
        print('正在加载学习率衰减方法……')
        if self.use_lr_scheduler == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer,
                                                         base_lr=1e-8,
                                                         max_lr=1e-3,
                                                         mode='triangular2',
                                                         cycle_momentum=False)
        else:
            raise ValueError(f'不支持{self.use_lr_scheduler}学习率衰减方法')
        print(f'学习率衰减方法{self.use_lr_scheduler}加载完成')

    def __epoch(self, now_epoch):
        print(f'开始Epoch[{now_epoch}/{self.train_epoch}]的训练……')
        all_loss = []
        for batch_id, (waveforms, labels) in enumerate(self.train_loader):
            self.total_train_step += 1
            self.optimizer.zero_grad()
            waveforms = waveforms.to(device=self.device)
            labels = labels.to(device=self.device)
            features = self.feature_extractor(waveforms)
            # identities = self.model.backbone(features)
            # outputs = self.model.softmax(identities)
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            # if labels.shape[0] % 2 == 1:
            #     identities = identities[0:-1]
            #     labels = labels[0:-1]
            # identities = torch.chunk(identities, 2, 0)
            # labels = torch.chunk(labels, 2, 0)
            # labels = torch.eq(labels[0], labels[1]).to(dtype=torch.int)
            # labels[labels == 0] = -1
            # loss += self.criterion(identities[0], identities[1], labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            all_loss.append(loss)
            if (batch_id + 1) % 100 == 0:
                print(
                    f'Epoch:[{now_epoch}/{self.train_epoch}], '
                    f'Batch:[{batch_id + 1}/{len(self.train_loader)}], '
                    f'Loss:{sum(all_loss) / len(all_loss):.4f}, '
                    f'Learning Rate: {self.scheduler.get_last_lr()[0]:.8f}'
                )
                all_loss = []
            self.writer.add_scalar(tag='train/loss',
                                   scalar_value=loss,
                                   global_step=self.total_train_step)
            self.writer.add_scalar(tag='train/learning_rate',
                                   scalar_value=self.scheduler.get_last_lr()[0],
                                   global_step=self.total_train_step)

    def __save_model(self, now_epoch):
        model_path = os.path.join(self.save_model_path, f'{self.use_model}/epoch_{now_epoch}')
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, f'{self.use_model}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        print('模型成功保存')

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
                features = self.feature_extractor(waveforms)
                identities = self.model.backbone(features)
                outputs = self.model.softmax(identities)
                acc.append(accuracy(outputs, labels))
                if labels.shape[0] % 2 == 1:
                    labels = labels[0:-1]
                    identities = identities[0:-1]
                identities = torch.chunk(identities, 2, dim=0)
                labels = list(torch.chunk(labels, 2, dim=0))
                score = F.cosine_similarity(identities[0], identities[1], dim=1)
                if all_score is None:
                    all_score = score
                else:
                    all_score = torch.cat([all_score, score], dim=0)
                if all_labels is None:
                    all_labels = labels
                else:
                    all_labels[0] = torch.cat([all_labels[0], labels[0]], dim=0)
                    all_labels[1] = torch.cat([all_labels[1], labels[1]], dim=0)
            print('验证集处理完毕')
            print('正在计算模型指标……')
            acc = sum(acc) / len(acc)
            fpr, fnr, all_score = fpr_fnr(all_score, all_labels)
            eer, threshold = eer_threshold(fpr, fnr, all_score)
            try:
                print(f'Speaker Identification ACC: {acc:.2%}\n'
                      f'Speaker Verification EER: {eer:.2%}, '
                      f'threshold: {threshold:.4f}')
            except:
                print('出现问题')
                print('acc', type(acc), acc)
                print('eer', type(eer), eer)
                print('threshold', type(threshold), threshold)
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
    # def evaluate(self):
    #     print('测试阶段开始……')
    #     eval_model = self.models.architecture
    #     eval_model.eval()
    #     dev_identities = None
    #     dev_labels = None
    #     test_identities = None
    #     test_labels = None
    #     with torch.no_grad():
    #         print('正在导入注册声纹特征……')
    #         for waveforms, labels in self.dev_loader:
    #             waveforms = waveforms.to(device=self.device)
    #             labels = labels.to(device=self.device)
    #             features = self.feature_extractor(waveforms)
    #             identities = eval_model(features)
    #             if dev_identities is None:
    #                 dev_identities = identities
    #             else:
    #                 dev_identities = torch.cat([dev_identities, identities], dim=0)
    #             if dev_labels is None:
    #                 dev_labels = labels
    #             else:
    #                 dev_labels = torch.cat([dev_labels, labels], dim=0)
    #         print('注册声纹特征导入完成')
    #         print(dev_identities.shape, dev_labels.shape)
    #         print('正在导入验证声纹特征……')
    #         for waveforms, labels in self.test_loader:
    #             waveforms = waveforms.to(device=self.device)
    #             labels = labels.to(device=self.device)
    #             features = self.feature_extractor(waveforms)
    #             identities = eval_model(features)
    #             if test_identities is None:
    #                 test_identities = identities
    #             else:
    #                 test_identities = torch.cat([test_identities, identities], dim=0)
    #             if test_labels is None:
    #                 test_labels = labels
    #             else:
    #                 test_labels = torch.cat([test_labels, labels], dim=0)
    #         print('验证声纹特征导入完成')
    #         print(test_identities.shape, test_labels.shape)
    #         print('正在计算模型指标……')
    #         # dev_labels = dev_labels.to(device='cpu')
    #         # test_labels = test_labels.to(device='cpu')
    #         # dev_identities = dev_identities.to(device='cpu')
    #         # test_identities = test_identities.to(device='cpu')
    #         score = F.cosine_similarity(dev_identities.unsqueeze(0), test_identities.unsqueeze(1), dim=2)
    #         score = score.reshape(-1)
    #         one_hot = F.one_hot(test_labels, len(dev_labels))
    #         one_hot = one_hot.reshape(-1)
    #         eer, threshold = eer_threshold(score, one_hot)
    #         return eer, threshold
