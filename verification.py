import os
import random

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audioF
from pydub import AudioSegment

from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.transform.feature_transform import feature_transform

if __name__ == "__main__":
    with torch.no_grad():
        path_1 = os.path.join('C:/Users/24667/Documents/录音/录音 (10).m4a')
        path_2 = os.path.join('C:/Users/24667/Documents/录音/录音 (7).m4a')
        target_sample_rate = 16000
        length = 3
        model_path = 'test/ECAPA_TDNN.pth'
        use_mono = True
        use_feature_extraction = 'MFCC'
        threshold = 0.6
        vad_triger = 3
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
                'time_masking': True,
                'time_mask_param': 5,
                'freq_masking': True,
                'freq_mask_param': 10,
                'iid_masks': True
            }
        }
        model_cfg = {
            'ECAPA_TDNN': {
                'channels': 1024,
                'bottleneck': 128
            }
        }

        audio_1 = AudioSegment.from_file(path_1)
        audio_2 = AudioSegment.from_file(path_2)
        audio_1_dir = os.path.join('./test', 'formatted_audio_1.wav')
        audio_2_dir = os.path.join('./test', 'formatted_audio_2.wav')
        audio_1.export(audio_1_dir, format='wav', bitrate='256k')
        audio_2.export(audio_2_dir, format='wav', bitrate='256k')
        audio_1, sample_rate_1 = torchaudio.load(uri=audio_1_dir)
        audio_2, sample_rate_2 = torchaudio.load(uri=audio_2_dir)

        if sample_rate_1 != target_sample_rate:
            audio_1 = audioF.resample(audio_1, sample_rate_1, target_sample_rate)
        if sample_rate_2 != target_sample_rate:
            audio_2 = audioF.resample(audio_2, sample_rate_2, target_sample_rate)

        audio_1 = audioF.vad(audio_1, target_sample_rate, trigger_level=vad_triger)
        audio_2 = audioF.vad(audio_2, target_sample_rate, trigger_level=vad_triger)

        assert audio_1.shape[1] != 0 and audio_2.shape[1] != 0, '音频长度为0，检查VAD阈值'

        if audio_1.shape[1] < audio_2.shape[1]:
            random_num = random.randint(0, audio_2.shape[1] - audio_1.shape[1] - 1)
            audio_2 = audio_2[:, random_num:random_num + audio_1.shape[1]]
        elif audio_1.shape[1] > audio_2.shape[1]:
            random_num = random.randint(0, audio_1.shape[1] - audio_2.shape[1] - 1)
            audio_1 = audio_1[:, random_num:random_num + audio_2.shape[1]]

        torchaudio.save(audio_1_dir, audio_1, target_sample_rate)
        torchaudio.save(audio_2_dir, audio_2, target_sample_rate)

        if use_mono:
            audio_1 = torch.mean(audio_1, dim=0, keepdim=False)
            audio_2 = torch.mean(audio_2, dim=0, keepdim=False)

        audio_1 = audio_1.unsqueeze(0)
        audio_2 = audio_2.unsqueeze(0)
        feature_extraction = feature_transform(use_feature_extraction, **feature_extraction_cfg)
        feature_1 = feature_extraction(audio_1)
        feature_2 = feature_extraction(audio_2)
        model = ECAPA_TDNN(**model_cfg['ECAPA_TDNN'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        output_1 = model(feature_1)
        output_2 = model(feature_2)
        score = F.cosine_similarity(output_1, output_2, dim=1).item()
        rate = (score + 1) / 2
        threshold_rate = (threshold + 1) / 2
        if score < threshold:
            print(f'\n  拒绝 ×\n')
        else:
            print(f'\n  通过 √\n')
        print(f'相似度: {rate:.2%}\n'
              f'当前判决门限: {threshold_rate:.2%}')
