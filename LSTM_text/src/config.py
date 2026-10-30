from pathlib import Path

import torch

# 当前项目的路径
ROOT_DIR = Path(__file__).parent.parent

# 原始数据文件的路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# 生成数据文件的路径
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
# 模型的文件路径
MODELS_DIR = ROOT_DIR / "models"
# 日志的文件路径
LOG_DIR = ROOT_DIR / "logs"


# 时间步的窗口大小
SEQ_LEN = 128
# 填充值的索引值
PAD_INDEX = 1

# 词向量的维度
EMBEDDING_DIM = 128
# 隐藏状态的维度
HIDDEN_DIM = 128

# 学习率
LEARNING_RATE = 0.001
# 训练的轮数
EPOCHS = 50

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
