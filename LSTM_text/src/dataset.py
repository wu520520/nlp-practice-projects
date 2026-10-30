# dataset.py 自定义的数据集
# 将数据转换成张量，然后进行批次处理
import torch
from torch.utils.data import Dataset, DataLoader

from day0731.src import config


class ReviewDataset(Dataset):

    # 初始化
    def __init__(self,file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            self.data = [eval(line) for line in f.readlines()]

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 循环遍历的方法
    def __getitem__(self, index):

        # 将data的数据，转换成input  target 张量
        d = self.data[index]

        input_tensor = torch.tensor(d["input"],dtype=torch.long)
        target_tensor = torch.tensor(d["target"],dtype=torch.float32)

        return input_tensor, target_tensor

    # 数据加载器
    @staticmethod
    def get_dataloader(file_path):
        # 创建数据集对象
        dataset = ReviewDataset(file_path)

        return DataLoader(dataset,config.SEQ_LEN,shuffle=True)

# if __name__ == '__main__':
#     loader = ReviewDataset.get_dataloader(config.PROCESSED_DATA_DIR/"train.jsonl")
#     print(len(loader))
