# train.py
# 加载数据，训练模型，并保存

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TransformerModel
from tokenizer import MyTokenizer


def train():
    # 加载数据
    train_loader = get_dataloader(config.PROCESSED_PATH / "train.jsonl")

    # 创建分词器，拿到词表大小
    my_tokenizer = MyTokenizer(config.PROCESSED_PATH / "vocab.txt")

    # 创建模型：传入词表大小和类别数
    model = TransformerModel(len(my_tokenizer.words), config.NUM_CLASSES)
    model.to(config.DEVICE)
    model.train()

    # 损失函数（多分类交叉熵）
    loss_function = torch.nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # TensorBoard 日志写入器（保存到 work4/logs）
    writer = SummaryWriter(log_dir=str(config.ROOT / 'logs'))

    min_loss = float('inf')
    for epoch in range(config.EPOCHS):
        total_loss = 0

        for batch_x, batch_y in tqdm(train_loader, desc='批次训练'):
            batch_x = batch_x.to(config.DEVICE)   # (batch, seq)
            batch_y = batch_y.to(config.DEVICE)   # (batch,)

            optimizer.zero_grad()

            output = model(batch_x)   # (batch, num_classes)
            loss = loss_function(output, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'第 {epoch + 1} 轮平均损失值: {avg_loss:.4f}')

        # 写入 TensorBoard 日志
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 损失更小时保存模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), str(config.MODELS_PATH / 'model.pt'))

    writer.close()


if __name__ == '__main__':
    train()
