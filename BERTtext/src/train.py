# train.py
# 训练模型
import torch
from torch import nn
from tqdm import tqdm

from BERTtext.src import config
from BERTtext.src.dataset import get_dataloader
from BERTtext.src.model import BertModel


def train():
    # 加载数据
    dataloader = get_dataloader(config.PROCESSED_PATH / "train.jsonl")

    # 创建模型
    model = BertModel()

    # 损失函数
    loss_function = nn.BCEWithLogitsLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr = config.LEARN_RATE
    )

    model.train()
    model.to(config.DEVICE)

    # 训练
    for epcho in range(config.EPOCHS):

        # 计算总损失值
        total_loss = 0
        # 计算最小损失值
        min_loss = float('inf')

        # 按照批次进行训练
        for input_list,attention_list,target_list in tqdm(dataloader,"批次训练"):
            # input_list [[],[],[]]
            input_list = input_list.to(config.DEVICE)

            # attention_list [[],[],[]]
            attention_list = attention_list.to(config.DEVICE)

            # target [,,,,,]
            target_list = target_list.to(config.DEVICE)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            output = model(input_list,attention_list)

            # 计算损失
            loss = loss_function(output, target_list)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 更新总损失值
            total_loss += loss.item()

        # 判断损失值是否减少，如果是减小了，则保存当前模型
        avg_loss = total_loss / len(dataloader)
        print(f"第{epcho+1}轮循环的平均损失值为:",avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            print("保存模型")

            torch.save(model.state_dict(), config.MODELS_PATH / "model.pt")

if __name__ == '__main__':
    train()

