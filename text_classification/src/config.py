# config.py
# 配置文件：定义项目路径、标签列表、训练超参数和设备
from pathlib import Path

import torch

# 项目根目录
ROOT = Path(__file__).parent.parent

# 数据目录
DATA_PATH = ROOT / "data"
# 原始数据路径
RAW_PATH = DATA_PATH
# 处理后数据保存路径
PROCESSED_PATH = DATA_PATH / "processed"
# 预训练模型 bert-base-chinese 的本地路径
PRETRAINED_PATH = r"G:\人工智能\P5\PYtest\BERT_model"
# 训练好的模型保存路径
MODELS_PATH = ROOT / "models"

# 10 个新闻分类标签
LABEL_LIST = ["教育", "科技", "房产", "体育", "社会", "娱乐", "游戏", "股票", "时政", "财经"]
# 分类数
NUM_CLASSES = len(LABEL_LIST)

# 序列最大长度
SEQ_LEN = 64
# 批次大小
BATCH_SIZE = 32
# 学习率
LEARN_RATE = 2e-5
# 训练轮次
EPOCHS = 3
# 训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
