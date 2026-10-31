from pathlib import Path

import torch

# 当前项目的根目录
ROOT = Path(__file__).parent.parent


# 输入文件的路径
RAW_PATH = ROOT / "data" / "raw"
# 生成文件的路径
PROCESSED_PATH = ROOT / "data" / "processed"
# 词表文件的路径
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.txt"
# 模型保存的路径
MODELS_PATH = ROOT / "model"


# 时间步长的窗口大小
SEQ_LEN = 128
# 批次大小
BATCH_SIZE = 32
# 学习率大小
LEARN_RATE = 0.0002
# 轮次大小
EPOCHS = 10

# 词向量维度
EMBED_DIM = 512
# Transformer头数
N_HEAD = 8
# Transformer编码/解码层数
NUM_LAYERS = 6
# 前馈网络维度
FFN_DIM = 2048
# Dropout概率
DROPOUT = 0.1

# 词表中特殊标记
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"  # 解码器起始标记

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
