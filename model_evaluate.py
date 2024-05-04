import os.path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.data.LibriSpeech_set import LibriSpeech_set
from utils.metrics import cal_fpr_fnr, cal_eer_threshold, cal_min_dcf_threshold
from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.transform.feature_transform import feature_transform


model_path = os.path.join('C:/Users/24667/Documents/毕设数据/2024_05_04_14_21_55/trained_models/',
                          '2024_05_04_14_21_55/ECAPA_TDNN/best_model/ECAPA_TDNN.pth')
use_feature_extraction = 'MFCC'
dataset_cfg = {
    'data_path': './dataset/LibriSpeech',
    'preprocess_cfg': {
        'max_duration': 3,
        'target_sample_rate': 16000,
        'keep_audio_channel': False,
        'use_speed_perturbation': False,
        'speed_perturbation_sequence': (0.9, 0.95, 1, 1.05, 1.1),
        'add_noise': False,
        'max_snr': 50,
        'min_snr': 10
    },
}
dataloader_cfg = {
    'batch_size': 128,
    'num_workers': 16,
    'pin_memory': True
}
feature_extraction_cfg = {
    'use_mfcc_cms': True,
    'feature_extraction_cfg': {
        'Spectrogram': {
            'n_fft': 1024,
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
    'Augmentations_cfg': {
        'time_masking': False,
        'time_mask_param': 5,
        'freq_masking': False,
        'freq_mask_param': 10,
        'iid_masks': True
    }
}
model_cfg = {
    'ECAPA_TDNN': {
        'channels': 1024,
        'bottleneck': 128,
        'scale': 8
    }
}


def set_data(_dataset_cfg, _dataloader_cfg):
    _test_set = LibriSpeech_set('test-other', **dataset_cfg)
    _test_loader = DataLoader(_test_set, shuffle=True, **dataloader_cfg)
    return _test_set, _test_loader


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    test_set, test_loader = set_data(dataset_cfg, dataloader_cfg)
    transform = feature_transform(use_feature_extraction, **feature_extraction_cfg)
    transform = transform.to(device=device)
    model = ECAPA_TDNN(**model_cfg['ECAPA_TDNN'])
    model = model.to(device=device)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    all_outputs = None
    all_labels = None
    all_identities = None
    print('模型评估开始……')
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device=device)
            labels = labels.to(device=device)
            features = transform(waveforms)
            identities = model(features)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

            if all_identities is None:
                all_identities = identities
            else:
                all_identities = torch.cat([all_identities, identities], dim=0)

        print('正在计算模型指标……')
        all_identities = F.normalize(all_identities, dim=-1)
        score_matrix = F.linear(all_identities, all_identities)
        labels_matrix = torch.eq(all_labels.unsqueeze(0), all_labels.unsqueeze(1))
        all_score = score_matrix.view(-1)
        all_labels = labels_matrix.view(-1)
        fpr, fnr, all_score = cal_fpr_fnr(all_score, all_labels)
        eer, eer_threshold = cal_eer_threshold(fpr, fnr, all_score)
        min_dcf, min_dcf_threshold = cal_min_dcf_threshold(fpr, fnr, all_score)
        print(f'EER: {eer:.4%}    '
              f'EER threshold: {eer_threshold:.8f}    '
              f'minDCF: {min_dcf:.8f}    '
              f'minDCF threshold: {min_dcf_threshold:.8f}')
