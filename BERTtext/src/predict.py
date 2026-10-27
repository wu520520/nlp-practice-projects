# predict.py
# 预测数据
import torch
from transformers import AutoTokenizer

from BERTtext.src import config
from BERTtext.src.model import BertModel

# 动态输入
while True:

    text = input("请输入评论：")

    if text.lower() == "exit" or text.lower() == "q":
        break

    # 转换成词向量
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_PATH)

    text_list = tokenizer(text,
                         padding="max_length",
                         max_length=config.SEQ_LEN,
                         truncation="longest_first")

    print(text_list)

    input_ids = torch.tensor([text_list["input_ids"]]).to(config.DEVICE)
    attention_mask = torch.tensor([text_list["attention_mask"]]).to(config.DEVICE)

    # 加载模型   训练完毕以后， 预训练+微调的模型
    model = BertModel()
    model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))

    # 预测结果
    model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():

        # 使用模型进行预测
        # 得到结果是一个小数
        output = model(input_ids, attention_mask)

        # 将这个小数，使用sigmoid进行处理，
        res = torch.sigmoid(output).item()

        # 并四舍五入
        comment = round(res)

        if comment == 1:
            print("好评",res)
        else:
            print("差评",(1-res))







    # 得到的结果是一个小数，
    # 将这个数值进行sigmoid计算，
    # 再根据四舍五入，转换成0-1的值



