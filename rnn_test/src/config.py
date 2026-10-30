 # config.py 配置文件，用于存放常量等配置数据

from pathlib import Path

import torch

# Path(__file__) 获取当前文件的绝对路径
# .parent 是获取当前路径的上一级目录
# 项目的根目录 绝对路径
ROOT_DIR = Path(__file__).parent.parent

# 原始数据的路径
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# 生成数据的路径
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# 模型的路径
MODELS_DIR = ROOT_DIR / "models"

# 日志的路径
LOGS_DIR = ROOT_DIR / "logs"

# 时间步的长度，窗口大小
SEQ_LEN = 5
# 批次大小
BATCH_SIZE = 1024
# 每个词的词向量的维度
EMBEDDING_DIM = 100
# 隐藏层的维度
HIDDEN_DIM = 100
# 学习率
LEARNING_RATE = 0.01
# 循环次数
EPOCHS = 1


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

