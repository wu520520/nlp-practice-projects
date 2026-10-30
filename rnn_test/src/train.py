#训练模型
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from day0729.rnn_test.src import config
from day0729.rnn_test.src.dataset import get_dataloader
from day0729.rnn_test.src.model import InputMethodModel
from day0729.rnn_test.src.tokenizer import JieBaTokenizer


def train_one_epoch(model,data_loader,loss_function, optimizer):
    # 用于计算平均损失
    total_loss = 0
    model.train()

    # 当前是只训练一轮
    for batch_x, batch_y in tqdm(data_loader, "按批次训练"):
        # 将训练数据，传入到gpu  cuda
        batch_x = batch_x.to(config.DEVICE)
        batch_y = batch_y.to(config.DEVICE)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        logits = model(batch_x)

        # 计算误差
        loss = loss_function(logits, batch_y)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    # 计算平均损失并返回
    return total_loss / len(data_loader)

# 多轮训练
def train():

    # 获取训练数据集
    dataloader = get_dataloader(config.PROCESSED_DATA_DIR/"train.jsonl")

    # 根据文件路径，获取words文件中的内容
    words = JieBaTokenizer.get_words_from_file(config.PROCESSED_DATA_DIR/"words.txt")

    # 创建模型
    model = InputMethodModel(len(words))

    model.to(config.DEVICE)

    # 设置损失函数
    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE
    )

    # 设置一个无穷大的值
    min_loss = float("inf")

    # 开启日志
    writer = SummaryWriter(config.LOGS_DIR)

    # 循环训练
    for epoch in tqdm(range(1,config.EPOCHS+1),"循环训练模型"):

        # 调用单轮训练
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer)

        # 记录日志
        writer.add_scalar("loss/epoch", avg_loss, epoch)

        if avg_loss < min_loss:
            # 说明当前的模型，比之前的要小
            # 直接保存
            torch.save(model.state_dict(),config.MODELS_DIR/"model.pth")

if __name__ == '__main__':
    train()