from pathlib import Path

import torch

# 设置路径
# 获得项目的根路径
ROOT = Path(__file__).parent

# 原始数据文件路径
RAW_PATH = ROOT / 'data' / 'raw'
PROCESSED_PATH = ROOT / 'data' / 'processed'
# 模型保存路径
MODELS_PATH = ROOT / 'models'


# 填充的索引
PAD_INDEX = 1
# sos索引
SOS_INDEX = 2
# eos索引
EOS_INDEX = 3


# 设置时间步长度
SEQ_LEN = 64
# 设置批次大小
BATCH_SIZE = 32
# 设置向量维度
EDIM = 128
# 设置轮次大小
EPOCHS = 30
# 设置学习率
LEARNING_RATE = 0.0001


# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
