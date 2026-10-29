# predict.py
# 预测：输入新闻标题，输出对应分类

import torch


import config
from model import TransformerModel
from tokenizer import MyTokenizer


def predict():
    while True:
        text = input('请输入新闻标题（输入 q 或 exit 退出）：')

        if text.lower() in ('q', 'exit'):
            break
        if not text.strip():
            print('输入不能为空，请重新输入')
            continue

        # 创建分词器并编码
        my_tokenizer = MyTokenizer(config.PROCESSED_PATH / "vocab.txt")
        input_list = torch.tensor([my_tokenizer.encode(text)],dtype = torch.long).to(config.DEVICE)
        # (1, seq_len)

        # 加载模型（并加载训练好的权重）
        model = TransformerModel(len(my_tokenizer.words),config.NUM_CLASSES)
        model.load_state_dict(torch.load(config.MODELS_PATH/'model.pt'))
        model.to(config.DEVICE)
        model.eval()

        with torch.no_grad():
            output = model(input_list)   # (1, num_classes)
            predict_index = output.argmax(-1).item()   # 取类别索引

        print('分类结果：', config.LABELS[predict_index])


if __name__ == '__main__':
    predict()
