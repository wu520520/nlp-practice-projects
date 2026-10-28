# dataset.py
# 数据集加载模块：读取 jsonl 文件，转换为张量，构建 DataLoader 批次数据
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
from text_classification.src import config


def get_dataloader(file_path):
    """
    读取处理后的 jsonl 文件，构造 DataLoader
    :param file_path: jsonl 文件路径
    :return: PyTorch DataLoader，按批次输出 (input_ids, attention_mask, target)
    """
    data_list = []
    # 使用 json.loads 安全解析（不使用 eval，避免代码注入风险）
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    input_list = []
    attention_list = []
    target_list = []

    # 拆分每条数据：input_ids、attention_mask、target
    for data in data_list:
        input_list.append(data["input_ids"])
        attention_list.append(data["attention_mask"])
        target_list.append(data["target"])

    # 转换为张量
    # input_ids 和 attention_mask 为整数索引，使用 long 类型
    input_list = torch.tensor(input_list, dtype=torch.long)
    attention_list = torch.tensor(attention_list, dtype=torch.long)
    # 多分类任务的目标值为类别索引（0-9），使用 long 类型（CrossEntropyLoss 要求）
    target_list = torch.tensor(target_list, dtype=torch.long)

    # 封装成 TensorDataset，再用 DataLoader 按批次输出
    dataset = TensorDataset(input_list, attention_list, target_list)

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
