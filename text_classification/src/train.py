# train.py
# 训练模块：使用训练集对 BERT 模型进行微调，并保存最优模型
import sys
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from text_classification.src import config
from text_classification.src.dataset import get_dataloader
from text_classification.src.model import BertModel


def train():
    # 1. 加载训练集数据
    dataloader = get_dataloader(config.PROCESSED_PATH / "train.jsonl")

    # 2. 创建模型（全参数微调）
    model = BertModel()

    # 3. 损失函数：多分类使用交叉熵损失（内含 softmax）
    #    与二分类使用 BCEWithLogitsLoss 不同
    loss_function = nn.CrossEntropyLoss()

    # 4. 优化器：Adam，学习率为 2e-5（BERT 微调推荐）
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config.LEARN_RATE
    )

    # 5. 切换到训练模式，并将模型迁移到设备
    model.train()
    model.to(config.DEVICE)

    # 最小损失初始值设为无穷大（在循环外初始化，避免每轮重置）
    min_loss = float('inf')

    # 6. 按 epoch 训练
    for epoch in range(config.EPOCHS):
        total_loss = 0       # 累计损失
        correct = 0          # 累计预测正确数
        total = 0            # 累计样本总数

        # 按批次训练，tqdm 显示进度条
        for input_list, attention_list, target_list in tqdm(dataloader, f"第{epoch+1}轮训练"):
            # 将数据迁移到设备
            input_list = input_list.to(config.DEVICE)
            attention_list = attention_list.to(config.DEVICE)
            target_list = target_list.to(config.DEVICE)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播，得到 logits
            output = model(input_list, attention_list)

            # 计算损失
            loss = loss_function(output, target_list)

            # 反向传播，计算梯度
            loss.backward()

            # 梯度裁剪，防止梯度爆炸（max_norm=1.0）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            # 累计损失
            total_loss += loss.item()

            # 计算准确率：取 logits 最大的类别作为预测结果
            pred = torch.argmax(output, dim=-1)
            correct += (pred == target_list).sum().item()
            total += len(target_list)

        # 本轮平均损失和准确率
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        # 使用 tqdm.write 输出，避免破坏进度条显示
        tqdm.write(f"第{epoch+1}轮: 损失={avg_loss:.4f}, 准确率={accuracy:.4f}")

        # 7. 如果当前损失更小，则保存模型
        if avg_loss < min_loss:
            min_loss = avg_loss
            config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODELS_PATH / "model.pt")
            tqdm.write(f"保存模型 (损失={avg_loss:.4f})")


if __name__ == '__main__':
    train()
