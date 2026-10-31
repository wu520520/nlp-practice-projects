# process.py
# 加载原始数据，并转换成词表，划分成训练集 测试集，并保存
import pandas as pd
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

from transformer import config
from transformer.tokenizer import MyTokenizer


def process():
    # 加载原始数据
    df = pd.read_csv(
        config.RAW_PATH / 'cmn.txt',
        sep='\t',
        names=["en","zh","other"],
        encoding='utf8'
    )

    print(df.info())

    # 分词处理 并保存词表
    # 处理中文分词
    MyTokenizer.build_vocab_zh(df,"zh")
    # 处理英文分词
    MyTokenizer.build_vocab_en(df, "en")

    # 设置映射   创建MyTokenizer对象，对象中包含 词表和映射字典
    # my_tokenizer = MyTokenizer.from_vocab("zh", "en")
    my_tokenizer = MyTokenizer(
        config.PROCESSED_PATH/"zh_vocab.txt",
        config.PROCESSED_PATH/"en_vocab.txt"
    )

    # 将词转换成索引  长度  填充
    input_list = []
    for line in df["zh"]:
        input_list.append(my_tokenizer.encode(line,False))

    target_list = []
    for line in df["en"]:
        target_list.append(my_tokenizer.encode(line,True))

    # 训练集  {input:[21],target:[22]}
    # 将input和target合并成 字典格式，并保存
    data_list = []
    for i in range(len(input_list)):
        data_list.append({"input":input_list[i], "target":target_list[i]})

    # 划分训练集和测试集
    data_train, data_test = train_test_split(
    data_list, test_size = 0.2, random_state = 42)

    # 保存训练集和测试集
    with open(config.PROCESSED_PATH/"train.jsonl","w",encoding="utf-8") as f:
        for data in data_train:
            f.write(str(data) + "\n")

    with open(config.PROCESSED_PATH / "test.jsonl", "w", encoding="utf-8") as f:
        for data in data_test:
            f.write(str(data) + "\n")

if __name__ == '__main__':
    process()


