# tokenizer.py
# 手动分词器：从原始文本构建字符级词表，并把文本转换成索引序列

import config


class MyTokenizer:

    # ================ 根据词表文件创建对象 ================
    def __init__(self, vocab_path):
        # TODO: 读取词表文件，每行去掉换行，得到词列表
        with open(vocab_path,'r',encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines()]

        # 保存词表
        self.words = words
        # 词 -> 索引  和  索引 -> 词 两个映射字典
        self.w2i = {w:i for i,w in enumerate(self.words)}
        self.i2w = {i:w for i,w in enumerate(self.words)}

    # ================ 从原始文本构建词表（字符级） ================
    @staticmethod
    def build_vocab(texts, save_path):
        # TODO: 遍历所有文本，收集出现的所有字
        # 中文按字切分
        chars = set()
        for text in texts:
            chars.update(text)

        # 词表前面固定放特殊 token
        vocab = ['<unk>', '<pad>'] + list(chars)

        # TODO: 把词表逐行写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            for w in vocab:
                f.write(w + "\n")

        return len(vocab)

    # ================ 文本转换成索引（截断 / 填充到 SEQ_LEN） ================
    def encode(self, text):
        # 未登录词用 <unk>(0) 代替
        indexes = [self.w2i.get(ch,config.UNK_INDEX) for ch in list(text)]

        if len(indexes) >= config.SEQ_LEN:
            # TODO: 超过长度就截断
            indexes = indexes[:config.SEQ_LEN]
        else:
            # TODO: 不足就填充 PAD
            indexes += [config.PAD_INDEX] * (config.SEQ_LEN - len(indexes))

        return indexes
