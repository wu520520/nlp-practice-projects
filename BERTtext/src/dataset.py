# dataset.py
# 批次 + 张量处理
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from BERTtext.src import config


def get_dataloader(file_path):

    data_list = []
    # 读取数据
    with open(file_path, 'rb') as f:
        data_list = [eval(line) for line in f.readlines()]

    # 拆分成input和target
    input_list = []
    attention_list = []
    target_list = []

    # data_list [
    # {"input_ids":[],"attention_mask":[],"target":n},
    # {"input_ids":[],"attention_mask":[],"target":n},
    # {"input_ids":[],"attention_mask":[],"target":n}]
    for i in range(len(data_list)):
        input_list.append(data_list[i]["input_ids"],)
        attention_list.append(data_list[i]["attention_mask"])
        target_list.append(data_list[i]["target"])

    # 结果中 input_list [[],[]]
    #  attention_list[[],[]]
    # target_list  [x,x,x,x,x,x]

    # 张量
    input_list = torch.tensor(input_list, dtype=torch.long)
    attention_list = torch.tensor(attention_list, dtype=torch.long)
    target_list = torch.tensor(target_list, dtype=torch.float)

    # 批次
    dataset = TensorDataset(input_list,attention_list, target_list)

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
