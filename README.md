# NLP 实践项目

自然语言处理（NLP）与深度学习序列模型实践集合，覆盖 **传统分词 → 词向量 → 文本分类 → 序列生成/翻译** 的完整进阶路线，包含 BERT / LSTM / RNN / GRU / Transformer / seq2seq 等主流模型。

## 📦 子项目

| 子项目 | 任务 | 技术 |
|--------|------|------|
| `BERTtext/` | BERT 中文文本分类（购物评论情感） | Transformers + BERT |
| `text_classification/` | 基于 BERT 预训练微调的文本分类 | Transformers + BERT |
| `text_classification_advanced/` | 自定义 tokenizer 的文本分类 | PyTorch 自建模型 |
| `LSTM_text/` | LSTM 文本分类 | PyTorch + LSTM |
| `rnn_test/` | RNN 文本分类 | PyTorch + RNN |
| `transformer/` | Transformer 中英翻译 | PyTorch + Transformer |

每个子项目均为标准结构：`src/` 代码 + `data/raw/` 原始数据 + `data/processed/` 处理产物（不入库）+ `models/` 权重（不入库）。

## 📜 独立学习脚本（仓库根目录）

- `day01.py` — jieba 中文分词
- `GRUtext.py` — GRU 微博评论情感分类
- `seq2seq_text.py` — seq2seq 序列翻译
- `TransformerTest.py` — Transformer 中英翻译
- `work0727.ipynb` — 综合练习

## 🛠 技术栈

Python | PyTorch | Transformers | BERT | LSTM | RNN | GRU | Transformer | seq2seq | jieba | scikit-learn

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 以某子项目为例（如 BERT 文本分类）
cd BERTtext
# 训练
python -m src.train
# 推理
python -m src.predict
```

## 📌 说明

- **模型权重与大数据集不入库**：BERT 完整权重、训练得到的 `.pt/.pth`、微博情感/词向量等公开大数据集均可自行下载，仓库仅提供代码与少量原始样例数据
- 各项目使用相对路径，克隆后可运行（需先准备 `data/raw/` 数据并按 `src/process.py` 生成处理产物）
