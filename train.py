import json
import os
from datetime import datetime

import torch

from utils.voiceprint_recognition import voiceprint_recognition

if __name__ == "__main__":
    timestr = datetime.now().strftime("%Y%m%d%H%M%S")
    train_cfg = {
        'use_gpu': True,
        'use_feature_extraction': 'MFCC',
        'use_model': 'ECAPA_TDNN',
        'use_veri_loss': 'MSELoss',
        'use_iden_loss': 'NLLLoss',
        'veri_iden_loss_weight': (1, 1),
        'use_optimizer': 'Adam',
        'use_lr_scheduler': 'CyclicLR',
        'data_path': './dataset/VoxCeleb1',
        'target_sample_rate': 16000,
        'train_epoch': 60,
        'logdir': os.path.join('./logs/' + timestr),
        'save_model_path': os.path.join('./trained_models/' + timestr)
    }
    dataset_cfg = {
        'data_path': './dataset/VoxCeleb1',
        'max_duration': 2,
        'use_mono': True,
        'audio_augument_cfg': {
            'use_speed_perturbation': True,
            'speed_perturbation_sequence': (0.9, 1, 1.1),
            'add_noise': True,
            'snr_range': (40, 60)
        }
    }
    dataloader_cfg = {
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True
    }
    transform_cfg = {
        'feature_extraction_cfg': {
            'Spectrogram': {
                'n_fft': 512 * 2,
                'win_length': 400,
                'hop_length': 160,
                'window_fn': torch.hamming_window,
                'normalized': True
            },
            'MFCC': {
                'n_mfcc': 80,
                'log_mels': True,
                'melkwargs': {
                    'n_fft': 400,
                    'hop_length': 160,
                    'n_mels': 80
                }
            }
        },
        'spec_augument_cfg': {
            'use_mfcc_cms': True,
            'time_masking': {
                'time_mask_param': 5,
                'iid_masks': True
            },
            'freq_masking': {
                'freq_mask_param': 10,
                'iid_masks': True
            },
        }
    }
    scheduler_cfg = {
        'base_lr': 1e-8,
        'max_lr': 1e-3,
        'step_size_up': 4000,
        'mode': 'triangular2',
        'cycle_momentum': False
    }
    voiceprint_recognition = voiceprint_recognition(**train_cfg,
                                                    dataset_cfg=dataset_cfg,
                                                    dataloader_cfg=dataloader_cfg,
                                                    transform_cfg=transform_cfg,
                                                    scheduler_cfg=scheduler_cfg)
    train_cfg = json.dumps(train_cfg, indent=4)
    dataset_cfg = json.dumps(dataset_cfg, indent=4)
    dataloader_cfg = json.dumps(dataloader_cfg, indent=4)
    transform_cfg = json.dumps(transform_cfg, indent=4)
    scheduler_cfg = json.dumps(scheduler_cfg, indent=4)
    voiceprint_recognition.train()
