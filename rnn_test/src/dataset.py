import torch
from torch.utils.data import Dataset, DataLoader

from day0729.rnn_test.src import config


class InputMethodDataset(Dataset):
    # 用于将数据生成张量，并进行批次处理
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引idx获取data中的数据，并转换成张量
        d = eval(self.data[idx])

        input_tensor = torch.tensor(d["input"],dtype=torch.long)
        target_tensor = torch.tensor(d["target"],dtype=torch.long)
        return input_tensor, target_tensor

# 用于直接加载数据文件，返回结果是一个数据加载器
# 已经将数据转换成张量，并且input和target都分开了的
def get_dataloader(file_path):
    dataset = InputMethodDataset(file_path)

    # 返回加载器
    return DataLoader(dataset,config.BATCH_SIZE,shuffle=True)

# if __name__ == '__main__':
#     # 对当前代码进行测试
#     loader = get_dataloader(config.PROCESSED_DATA_DIR/"train.jsonl")
#
#     for batch_x,batch_y in loader:
#         print(batch_x.shape, batch_y.shape)

