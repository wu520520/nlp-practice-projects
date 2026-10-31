# predict.py
# 预测数据
import torch

import config
from model import TransformerModel
from process import load_vocab, encode

# 动态输入
with open(config.PROCESSED_PATH / 'train.txt', "r", encoding="utf-8") as f:
    data_list = [line.replace("\n", "").split(',') for line in f.readlines()]

while True:

    text = input("请输入新闻：")

    if text.lower() == "exit" or text.lower() == "q":
        break

    # 加载词表
    vocab = load_vocab()

    # 将文本编码为input_ids和attention_mask
    input_ids, attention_mask = encode(text, vocab)

    input_ids = torch.tensor([input_ids]).to(config.DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(config.DEVICE)

    # 加载模型
    model = TransformerModel()
    model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))

    # 预测结果
    model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():

        # 使用模型进行预测
        # 得到结果是一个小数
        output = model(input_ids, attention_mask)

        pred = torch.argmax(output, dim=-1).item()

        for line in data_list:
            if int(line[0]) == pred:
                print(f"预测类别索引：{line[1]}")
                break
