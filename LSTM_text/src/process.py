# process.py
# 处理数据

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from day0731.src import config
from day0731.src.tokenizer import JieBaTokenizer


def process():
    # 加载数据
    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv",
                     usecols=["review","label"],
                     encoding='utf-8')

    # 清洗数据
    df.dropna(inplace=True)
    df = df[df["review"].str.strip().ne("")]

    # 拆分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 进行分词
    # 构建词表
    JieBaTokenizer.build_vocab(
           df["review"].tolist(),
           config.PROCESSED_DATA_DIR / "words.txt"
    )

    # 根据路径加载词表文件
    tokenizer = JieBaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "words.txt")

    # 将训练集进行转换，并保存
    train_list = get_list_from_df(tokenizer,train_df)

    with open(config.PROCESSED_DATA_DIR / "train.jsonl", "w") as f:
        for d in train_list:
            f.write(str(d) + "\n")

    # 将测试集进行转换，并保存
    test_list = get_list_from_df(tokenizer, test_df)
    with open(config.PROCESSED_DATA_DIR / "test.jsonl", "w") as f:
        for d in test_list:
            f.write(str(d) + "\n")


# 将传入的df，进行切分，review转换成索引list
# 返回值是[{"input":[数值],"target":数值}]
def get_list_from_df(tokenizer,df):
    input_list = []
    target_list = df["label"].tolist()

    # 将input中的评论，进行分词，并转换成索引列表
    for review in tqdm(df["review"].tolist(),"转换review为索引列表"):
        input_list.append(tokenizer.encode(review))

    # 将train的input和target整合
    list = []

    for i in range(len(input_list)):
        list.append(
            {
                "input": input_list[i],
                "target": target_list[i]
            }
        )
    return list

if __name__ == '__main__':
    process()



