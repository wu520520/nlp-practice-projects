# train.py 训练模型
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from day0731.src import config
from day0731.src.dataset import ReviewDataset
from day0731.src.model import ReviewAnalyzeModel
from day0731.src.tokenizer import JieBaTokenizer


def train():
    # 加载数据
    loader = ReviewDataset.get_dataloader(config.PROCESSED_DATA_DIR/"train.jsonl")

    # 获取词表文件中所有的词的个数
    tokenizer = JieBaTokenizer.from_vocab(config.PROCESSED_DATA_DIR/"words.txt")

    # 创建模型
    model = ReviewAnalyzeModel(tokenizer.words_size)

    # 损失函数
    loss_function = nn.BCEWithLogitsLoss()

    # 优化器
    optim = torch.optim.Adam(
        model.parameters(),
        lr = config.LEARNING_RATE
    )

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    min_loss = float("inf")
    # 批次训练
    for epoch in range(config.EPOCHS):
        # 开始训练
        avg_loss = train_one_epoch(model, loader, loss_function,optim)
        print("epoch:",epoch,"avg_loss:",avg_loss)

        writer.add_scalar("loss/epoch", avg_loss, epoch)

        # 将模型进行保存
        if avg_loss < min_loss:
            min_loss = avg_loss
            # 保存
            torch.save(model.state_dict(),config.MODELS_DIR / "model.pth")


# 训练一次模型
def train_one_epoch(model,loader,loss_function,optim):

    model.to(config.DEVICE)

    model.train()

    # 用于计算平均损失
    total_loss = 0

    for batch_x, batch_y in tqdm(loader,"批次处理"):
        batch_x = batch_x.to(config.DEVICE)
        batch_y = batch_y.to(config.DEVICE)

        # 清空梯度
        optim.zero_grad()

        output = model(batch_x)

        # 计算损失
        loss = loss_function(output, batch_y)

        # 反向传播
        loss.backward()

        # 更新参数
        optim.step()

        total_loss += loss.item()

    # 计算平均损失，并返回
    return total_loss / len(loader)


if __name__ == '__main__':
    # 训练模型
    train()



