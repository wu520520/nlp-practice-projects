# process.py
# 处理数据
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import config


def build_vocab(text_list):
    """根据训练文本构建字级词表"""
    counter = Counter()
    for text in tqdm(text_list, desc="构建词表"):
        # 使用字级分词 (char-level)
        tokens = list(text)
        counter.update(tokens)

    # 词表中预留特殊标记：[PAD]=0, [UNK]=1, [BOS]=2(解码器起始标记)
    vocab = {config.PAD_TOKEN: 0, config.UNK_TOKEN: 1, config.BOS_TOKEN: 2}
    # 限制词表大小，只保留前 5000 个高频字，剩余字映射为 [UNK]
    for idx, (word, _) in enumerate(counter.most_common(5000), start=len(vocab)):
        vocab[word] = idx

    # 保存词表
    with open(config.VOCAB_PATH, "w", encoding="utf-8") as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")

    return vocab


def load_vocab():
    """加载词表"""
    vocab = {}
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word, idx = line.replace("\n", "").split("\t")
            vocab[word] = int(idx)
    return vocab


def encode(text, vocab):
    """将文本编码为input_ids (字级，不加特殊标记)"""
    input_ids = []
    for token in list(text):  # 字级分词
        input_ids.append(vocab.get(token, vocab[config.UNK_TOKEN]))

    # 截断
    if len(input_ids) > config.SEQ_LEN:
        input_ids = input_ids[:config.SEQ_LEN]
    # 填充
    while len(input_ids) < config.SEQ_LEN:
        input_ids.append(vocab[config.PAD_TOKEN])

    # attention_mask: 真实token为1，padding为0
    attention_mask = [1 if idx != vocab[config.PAD_TOKEN] else 0 for idx in input_ids]
    return input_ids, attention_mask


def process(path, vocab=None):
    """处理数据并保存"""
    # 加载数据
    with open(config.RAW_PATH / (path + '.txt'), "r", encoding="utf-8") as f:
        data_list = [line.replace("\n", "").split('\t') for line in f.readlines()]
        df = pd.DataFrame(data_list, columns=["news", "label_str"])
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label_str"])

    df_label = df.drop_duplicates(subset=["label"], keep="first")
    print(df_label)

    # 处理训练数据时构建词表
    if vocab is None:
        vocab = build_vocab(df["news"].tolist())

    # 对数据进行分词与编码
    data_list = []
    for i in tqdm(range(len(df)), desc="处理数据"):
        input_ids, attention_mask = encode(df["news"].iloc[i], vocab)
        d = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": int(df["label"].iloc[i])
        }
        data_list.append(d)

    # 保存处理后的数据
    with open(config.PROCESSED_PATH / (path + ".jsonl"), "w", encoding="utf-8") as f:
        for data in data_list:
            f.write(str(data) + "\n")

    # 保存标签映射 {索引:标签名}
    with open(config.PROCESSED_PATH / (path + ".txt"), "w", encoding="utf-8") as f:
        for i, data in enumerate(df_label["label"].tolist()):
            f.write(str(data) + "," + str(df_label["label_str"].tolist()[i]) + "\n")

    return vocab


if __name__ == "__main__":
    # 先处理训练集，构建词表
    vocab = process('train')
    # 测试集复用训练集的词表
    process('test', vocab)
