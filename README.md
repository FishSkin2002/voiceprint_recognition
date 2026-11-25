# 声纹识别（Voiceprint Recognition）

这是一个基于 PyTorch 和 TorchAudio 的说话人识别与验证项目。模型从语音中提取声纹向量，用于判断两段录音是否来自同一位说话人。项目包含 VoxCeleb1 训练、LibriSpeech 批量评估和双音频验证流程，提供 ECAPA-TDNN 与 ERes2Net 两种模型。

## 项目目录

```text
.
├── train.py                    # 训练与验证
├── model_evaluate.py           # 批量评估并计算验证指标
├── verification.py             # 验证两段录音是否来自同一说话人
├── requirements.txt            # Python 依赖
└── utils/
    ├── voiceprint_recognition.py  # 训练循环、验证和权重保存
    ├── metrics.py                 # 准确率、EER 和 minDCF
    ├── data/                      # 数据集读取与波形预处理
    ├── models/                    # ECAPA-TDNN、3D CNN 和分类头
    ├── download/                  # ERes2Net 及其融合和池化模块
    ├── nn/                        # 损失函数、激活函数和池化层
    └── transform/                 # 特征提取与特征增强
```

## 环境准备

在项目根目录中运行以下命令，安装 CUDA 11.8 版的 PyTorch 和 TorchAudio 及其余依赖。`requirements.txt` 中的 `~=2.2.1+cu118` 版本写法无法被 pip 解析，下面的命令直接指定了对应版本。

```bash
python -m pip install torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "torchinfo~=1.8.0" soundfile tensorboard "pydub~=0.25.1" "tqdm~=4.66.2"
```

仅使用 CPU 时，将第一条命令中的 `cu118` 替换为 `cpu`，并在 `train.py` 中将 `train_cfg['use_gpu']` 设为 `False`。验证 M4A 等需要转码的音频时，还需安装 FFmpeg 并将其加入 `PATH`。[PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)

## 主要脚本用法

### 1. 训练模型 `train.py`

将 VoxCeleb1 音频解压到 `dataset/VoxCeleb1/wav/`，按数据集原有的说话人和视频目录存放，并将 `iden_split.txt` 放到 `dataset/VoxCeleb1/`。如果数据放在其他位置，修改 `dataset_cfg['data_path']`。加载器使用划分文件中的 `1` 和 `2` 分别读取训练集和验证集。

默认配置使用 16 kHz 单声道音频、80 维 MFCC、ECAPA-TDNN 和 AAMLoss。可以在 `train_cfg`、`model_cfg` 和 `dataloader_cfg` 中调整模型、训练轮数和批大小。

```bash
python train.py
```

训练日志写入 `logs/<时间戳>/<模型名>/`，权重写入 `trained_models/<时间戳>/<模型名>/`。验证集 EER 最低的权重保存在 `best_model/<模型名>.pth`，各轮验证结果记录在 `metrics.txt`。

### 2. 批量评估 `model_evaluate.py`

将 LibriSpeech `test-other` 解压到 `dataset/LibriSpeech/test-other/`，或修改 `dataset_cfg['data_path']` 指向包含 `test-other/` 的数据根目录。将 `model_path` 改为自己的权重路径。

脚本默认使用 ERes2Net 和 MFCC。如果评估默认训练产出的 ECAPA-TDNN 权重，将模型初始化改为脚本中已有的 ECAPA-TDNN 写法，保留模型移到 `device` 的步骤，并将 `feature_extraction_cfg['feature_transpose']` 设为 `False`。模型结构和特征配置应与训练时一致。仅用 CPU 读取 CUDA 权重时，为 `torch.load` 指定 `map_location=device`。

```bash
python model_evaluate.py
```

脚本使用声纹向量的余弦相似度计算等错误率（EER）、最小检测代价（minDCF）和对应阈值，将结果打印到终端并写入 `log_dir` 指定的 TensorBoard 日志目录。

### 3. 验证两段录音 `verification.py`

在项目根目录创建 `test/`，并在脚本中修改 `path_1`、`path_2` 和 `model_path`，分别指向两段录音和 ECAPA-TDNN 权重。脚本将中间 WAV 文件写入 `test/`。

验证脚本示例使用 1024 通道的 ECAPA-TDNN。如果加载 `train.py` 默认训练产出的权重，将 `model_cfg['ECAPA_TDNN']['channels']` 改为 `512`，并为 `torch.load` 指定 `map_location='cpu'`。可以修改 `threshold` 调整余弦相似度的判定阈值，默认值为 `0.6`。

```bash
python verification.py
```

脚本输出验证通过或拒绝，以及归一化后的相似度和判定阈值。
