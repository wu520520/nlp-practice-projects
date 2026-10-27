# process.py
# 处理数据
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from BERTtext.src import config

# 加载数据
df = pd.read_csv(
    config.RAW_PATH / "online_shopping_10_cats.csv",
    encoding="utf-8",
)

df.dropna(inplace=True)
print(df.isnull().sum())

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_PATH)

# 对数据进行分词
tokenizer_result = tokenizer(df["review"].tolist(),
          padding="max_length",
          max_length=config.SEQ_LEN,
          truncation="longest_first",
          )
# 提取结果中的 input_ids    attention_mask
# 整合成一个list
# data_list = [{"input_ids":[],"attention_mask":[],"target":n},{},{} ]

data_list = []
for i in tqdm(range(len(df)),"处理数据"):
    d = {
        "input_ids":tokenizer_result["input_ids"][i],
        "attention_mask":tokenizer_result["attention_mask"][i],
        "target":df["label"].iloc[i]
    }

    data_list.append(d)

print(len(data_list))

# 划分训练集 测试集
data_train, data_test = (
    train_test_split(data_list, test_size=0.2,shuffle=True))
# 1  0.9    0  0.1
# 保存文件
with open(config.PROCESSED_PATH / "train.jsonl","w",encoding="utf-8") as f:
    for data in data_train:
        f.write(str(data)+ "\n")

with open(config.PROCESSED_PATH / "test.jsonl","w",encoding="utf-8") as f:
    for data in data_test:
        f.write(str(data)+ "\n")


