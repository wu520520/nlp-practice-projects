# train.py
# 加载数据，训练模型，并进行保存

import torch
from tqdm import tqdm

from transformer import config
from transformer.dataset import get_dataloader
from transformer.model import TransformerModel
from transformer.tokenizer import MyTokenizer


def train():
    # 加载数据
    train_loader = get_dataloader(config.PROCESSED_PATH/"train.jsonl")

    # 创建分词器，需要用到对应的词表的长度
    my_tokenizer = MyTokenizer(
        config.PROCESSED_PATH/"zh_vocab.txt",
        config.PROCESSED_PATH/"en_vocab.txt"
    )

    # 创建模型 需要传入 中文词表大小  英文词表大小
    model = TransformerModel(len(my_tokenizer.zh_words), len(my_tokenizer.en_words))

    # 开启训练模式
    model.to(config.DEVICE)
    model.train()

    # 损失函数
    loss_function = (
        torch.nn.CrossEntropyLoss(ignore_index=config.PAD_INDEX))

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE)


    # 按轮次运行
    for epoch in range(config.EPOCHS):

        # 每个轮次的总损失
        total_loss = 0
        min_loss = float('inf')

        # 按批次训练模型
        for batch_x,batch_y in tqdm(train_loader,"批次训练"):
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)

            # 设置清空梯度
            optimizer.zero_grad()

            # 处理batch_y，拆分成输入和输出
            en_input = batch_y[:,:-1]
            en_target = batch_y[:,1:]

            # 前向传播
            output = model(batch_x, en_input)

            # 一条数据   输入 63个    [1,63,128]
            # 输出 [1,63,7971]    [63,7971]
            # 真实值 63 [1,63]   en_target.reshape(-1)  63

            # 计算损失值
            loss = loss_function(
                output.reshape(-1,len(my_tokenizer.en_words)),
                en_target.reshape(-1)
            )

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 比较平均损失，将最小的保存
            total_loss += loss.item()

        # 计算平均值
        avg_loss = total_loss / len(train_loader)
        print(f"第{epoch + 1}次循环的平均损失值为:",avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            print("保存模型")

            torch.save(model.state_dict(),config.MODELS_PATH / "model.pt")



if __name__ == '__main__':
    train()
