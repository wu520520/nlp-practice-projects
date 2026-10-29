# dataset.py
# 加载处理后的 jsonl 数据，转换成张量并封装成批次

import torch
from torch.utils.data import TensorDataset, DataLoader

import config


def get_dataloader(file_path):
    # data_list 中每条是 {'input': [索引...], 'target': 类别索引}
    # TODO: 读取文件，每行用 eval 解析成 dict，装进列表
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = [eval(line) for line in f.readlines()]

    # TODO: 从 data_list 里拆出 input 和 target 两个列表
    input_list = [data["input"] for data in data_list]
    target_list = [data["target"] for data in data_list]

    # 转成张量
    input_list = torch.tensor(input_list, dtype=torch.long)    # (N, SEQ_LEN)
    target_list = torch.tensor(target_list, dtype=torch.long)  # (N,)

    # 封装成批次
    dataset = TensorDataset(input_list, target_list)
    return DataLoader(dataset, config.BATCH_SIZE, shuffle=True)


# 数据加载测试入口
if __name__ == '__main__':
    # TODO: 取一个 batch 打印 input / target 的形状
    loader = get_dataloader(config.PROCESSED_PATH / "train.jsonl")
    for x, y in loader:
        print("input 形状:", x.shape, " target 形状:", y.shape)
        # input 形状: torch.Size([32, 64])  target 形状: torch.Size([32])
