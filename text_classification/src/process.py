# process.py
# 数据处理模块：读取原始 txt 数据 → 分词 → 保存为 jsonl
import json


from tqdm import tqdm
from transformers import AutoTokenizer

from text_classification.src import config
# 加载 bert-base-chinese 的分词器
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_PATH)

# 构建标签与 id 的双向映射字典
label2id = {label: idx for idx, label in enumerate(config.LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(config.LABEL_LIST)}


def load_txt(file_path):
    """
    读取原始 txt 数据文件
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                title = parts[0]
                label = parts[-1]
                data.append((title, label))
    return data


def tokenize_data(data, desc="处理数据"):
    """
    对文本数据进行分词处理
    :param data: [(标题, 分类), ...]
    :param desc: 进度条描述
    :return: [{"input_ids": [], "attention_mask": [], "target": 类别id}, ...]
    """
    data_list = []
    for title, label in tqdm(data, desc):
        # 使用分词器进行分词：padding 到最大长度、过长截断
        tokenized = tokenizer(
            title,
            padding="max_length",
            max_length=config.SEQ_LEN,
            truncation="longest_first"
        )
        d = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            # 将字符串标签转换为整数 id
            "target": label2id.get(label, 0)
        }
        data_list.append(d)
    return data_list


def save_jsonl(data_list, save_path):
    """
    将处理后的数据保存为 jsonl 文件（每行一个 json 对象）
    使用 json.dumps 而非 str()，确保格式正确且安全
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    # 1. 加载原始 txt 数据
    train_data = load_txt(config.RAW_PATH / "train.txt")
    test_data = load_txt(config.RAW_PATH / "test.txt")

    # 2. 对文本进行分词处理（使用全部数据，不做采样）
    train_tokenized = tokenize_data(train_data, "处理训练数据")
    test_tokenized = tokenize_data(test_data, "处理测试数据")

    # 3. 确保保存目录存在
    config.PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # 4. 保存处理后的数据为 jsonl 文件
    save_jsonl(train_tokenized, config.PROCESSED_PATH / "train.jsonl")
    save_jsonl(test_tokenized, config.PROCESSED_PATH / "test.jsonl")

    # 5. 保存标签映射字典
    with open(config.PROCESSED_PATH / "label_map.json", 'w', encoding='utf-8') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    print('处理后的数据已保存至 G:\\人工智能\\P5\\PYtest\\final_exam\\data\\processed')


if __name__ == '__main__':
    main()
