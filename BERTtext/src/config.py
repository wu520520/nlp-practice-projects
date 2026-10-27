from pathlib import Path

import torch

# 当前项目的根目录
ROOT = Path(__file__).parent.parent

print(ROOT)
# 输入文件的路径
RAW_PATH = ROOT / "data" / "raw"
# 生成文件的路径
PROCESSED_PATH = ROOT / "data" / "processed"
# 预训练模型的路径
PRETRAINED_PATH = r"G:\人工智能\P5\PYtest\BERT_model"
# 模型保存的路径
MODELS_PATH = ROOT / "models"


# 时间步长的窗口大小
SEQ_LEN = 64
# 批次大小
BATCH_SIZE = 32
# 学习率大小
LEARN_RATE = 0.0001
# 轮次大小
EPOCHS = 10

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
