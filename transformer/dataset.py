# dataset.py
# 加载训练集 测试集的数据文件，并转换成张量和批次处理
import torch
from torch.utils.data import TensorDataset, DataLoader

from transformer import config


def get_dataloader(file_path):
    # 读取数据，转换成张量，设置批次并返回批次
    data_list = []
    with open(file_path,"r",encoding="utf-8") as f:
        data_list = [eval(line) for line in f.readlines()]

    # 数据结构[{"input":[12],"target":[2134]},{},{}]
    input_list = [data["input"] for data in data_list ]
    target_list = [data["target"] for data in data_list]

    # 张量
    input_list = torch.tensor(input_list,dtype=torch.long)
    target_list = torch.tensor(target_list, dtype=torch.long)

    # 设置批次
    dataset = TensorDataset(input_list,target_list)
    loader = DataLoader(dataset,config.BATCH_SIZE,shuffle=True)
    return loader
