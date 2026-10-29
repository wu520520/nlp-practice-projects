# process.py
# 数据预处理：读取原始 txt、构建词表、编码并保存为 jsonl

import config
from tokenizer import MyTokenizer


def read_txt(file_path):
    """读取 tab 分隔的「标题\t标签」文本，返回 (文本列表, 标签列表)"""
    # TODO: 打开文件，逐行按 \t 切分，标题和标签分别装进两个列表
    texts, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            parts = line.split("\t")
            texts.append(parts[0].strip())
            labels.append(parts[-1].strip())
    return texts, labels


def process():
    # 读取原始数据（题目已给好 train / test 划分）
    train_texts, train_labels = read_txt(config.RAW_PATH / 'train.txt')
    test_texts, test_labels = read_txt(config.RAW_PATH / 'test.txt')

    # 用全部文本构建词表（保证测试集的字也能覆盖到）
    all_texts = train_texts + test_texts
    MyTokenizer.build_vocab(all_texts, config.PROCESSED_PATH / 'vocab.txt')

    # 创建分词器对象
    my_tokenizer = MyTokenizer(config.PROCESSED_PATH / 'vocab.txt')

    # 标签名 -> 索引
    label2id = {name:idx for idx, name in enumerate(config.LABELS)}

    # 分别处理 train 和 test
    for split, texts, labels in [
        ('train', train_texts, train_labels),
        ('test', test_texts, test_labels),
    ]:
        data_list = []
        for text, label in zip(texts, labels):
            data_list.append({
                'input': my_tokenizer.encode(text),   # 编码后的索引序列
                'target': label2id[label],  # 类别索引
            })

        # TODO: 把 data_list 逐行（str(dict)）写入 jsonl 文件
        with open(config.PROCESSED_PATH / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for data in data_list:
                f.write(str(data) + "\n")

        print(f'{split} 处理完成，共 {len(data_list)} 条')


if __name__ == '__main__':
    process()
