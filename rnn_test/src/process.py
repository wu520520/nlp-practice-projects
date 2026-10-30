# process.py 用于对数据进行预处理
import json

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from day0729.rnn_test.src.tokenizer import JieBaTokenizer


# 定义函数或者类
def process():

    # 加载数据 lines=True 设置成一行数据，识别成一个json数据
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", encoding="utf-8",lines=True)
    print(df)

    # 获取dialog中的对话内容，去掉 user1： 和user2：
    sentences = []
    for li in tqdm(df["dialog"], desc="获取dialog中的对话内容"):
        for line in li:
            sentences.append(line.split("：")[1])

    # 直接调用分词器中的build_vocab方法
    # 将所有的文字分词，并存入到词表文件中
    JieBaTokenizer.build_vocab(sentences,config.PROCESSED_DATA_DIR / "words.txt")

    tizer = JieBaTokenizer(sentences)
    sent_idx = tizer.encode(sentences)

    # 处理input 和target
    input_list = []
    target_list = []

    for sent in tqdm(sent_idx, desc="处理input和target数据"):
        for i in range(len(sent) - config.SEQ_LEN):
            # 将前5个值存入到input中
            input_list.append(sent[i:i + config.SEQ_LEN])
            # 将第6个值，存入到target中
            target_list.append(sent[i + config.SEQ_LEN])

    # 将数据进行拆分，并存入到文件中，训练文件和测试文件
    # 使用json格式，每一条数据，都包含 input和target
    datas_list = []
    for i in range(len(input_list)):
        # 将每一组元素，都存入一个单独的字典中
        datas = {}
        datas["input"] = input_list[i]
        datas["target"] = target_list[i]
        datas_list.append(datas)

    # 进行数据拆分 8:2
    train_data,test_data = (
        train_test_split(datas_list, test_size = 0.2, random_state = 42))

    # 分别将训练数据和测试数据进行保存
    with open(config.PROCESSED_DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for data in tqdm(train_data,"保存训练集数据"):
            f.write(str(data)+ "\n")

    with open(config.PROCESSED_DATA_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for data in tqdm(test_data,"保存测试集数据"):
            f.write(str(data)+ "\n")


if __name__ == '__main__':
    # 读取文件，处理数据，进行分词，并保存到文件中
    process()
