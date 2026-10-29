# config.py
# 项目配置文件：统一管理路径与超参数

from pathlib import Path

import torch

# 项目根目录
ROOT = Path(__file__).parent.parent

# ===== 路径设置 =====
RAW_PATH = ROOT / 'data' / 'raw'           # 原始数据（train.txt / test.txt）
PROCESSED_PATH = ROOT / 'data' / 'processed'  # 处理后数据（vocab.txt / train.jsonl / test.jsonl）
MODELS_PATH = ROOT / 'models'             # 模型保存路径

# ===== 特殊 token 索引 =====
UNK_INDEX = 0  # 未登录词
PAD_INDEX = 1  # 填充

# ===== 新闻分类标签 =====
# 共 10 个类别，列表下标即类别索引（标签名 -> 0~9 的数字）
LABELS = ['体育', '娱乐', '房产', '教育', '时政', '游戏', '社会', '科技', '股票', '财经']
NUM_CLASSES = len(LABELS)

# ===== 训练超参数 =====
SEQ_LEN = 64          # 最大序列长度
BATCH_SIZE = 32       # 批处理大小
EDIM = 128             # 词向量维度
NHEAD = 8            # 多头注意力头数
NUM_LAYERS = 3      # Transformer 编码器层数
EPOCHS = 10           # 训练轮数
LEARNING_RATE = 0.0001    # 学习率

# 运行设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
