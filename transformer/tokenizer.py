from nltk import word_tokenize

from transformer import config


class MyTokenizer():

    # ================根据词表文件创建对象================
    def __init__(self,zh_path, en_path):

        with open(zh_path, "r", encoding="utf-8") as f:
            zh_words = [line.strip("\n") for line in f.readlines()]

        with open(en_path, "r", encoding="utf-8") as f:
            en_words = [line.strip("\n") for line in f.readlines()]

        self.zh_words = zh_words
        self.en_words = en_words

        # 映射字典
        zh_i2w = {}
        zh_w2i = {}

        for i in range(len(zh_words)):
            zh_i2w[i] = zh_words[i]
            zh_w2i[zh_words[i]] = i

        en_i2w = {}
        en_w2i = {}

        for i in range(len(en_words)):
            en_i2w[i] = en_words[i]
            en_w2i[en_words[i]] = i

        self.zh_i2w = zh_i2w
        self.zh_w2i = zh_w2i
        self.en_w2i = en_w2i
        self.en_i2w = en_i2w

    # @staticmethod
    # def from_vocab():
    #
    #     return MyTokenizer(zh_words,en_words)

    #================原始文件创建词表================
    @staticmethod
    def build_vocab_en(df,col):
        en_words = set()
        for line in df[col]:
            en_words.update(word_tokenize(line))

        en_words = ["<unk>", "<pad>", "<sos>", "<eos>"] + list(en_words)

        # 保存
        with open(config.PROCESSED_PATH/"en_vocab.txt","w",encoding="utf-8") as f:
            for word in en_words:
                f.write(word + "\n")

    @staticmethod
    def build_vocab_zh(df,col):
        zh_words = set()
        for line in df[col]:
            zh_words.update(list(line))

        zh_words = ["<unk>", "<pad>", "<sos>", "<eos>"] + list(zh_words)

        # 保存
        with open(config.PROCESSED_PATH/"zh_vocab.txt","w",encoding="utf-8") as f:
            for word in zh_words:
                f.write(word + "\n")

    # ================将原始数据转换成索引================
    def encode(self,text,has_sos):
        # text 是一条数据
        if has_sos:
            # 如果有sos，说明是英文的
            indexes = [self.en_w2i[word] for word in word_tokenize(text)]

            # 判断长度，超过 SEQ_LEN-2的，则截取，不足的填充
            # 需要添加上<sos><eos>
            if len(indexes) >= config.SEQ_LEN - 2:
                indexes = [config.SOS_INDEX] + indexes[:config.SEQ_LEN - 2] + [config.EOS_INDEX]
            else:
                indexes = [config.SOS_INDEX] + indexes + [config.EOS_INDEX] + ([config.PAD_INDEX] * (config.SEQ_LEN - 2 - len(indexes)))

            return indexes
        else:
            # 如果没有sos，说明是中文的
            indexes = [self.zh_w2i[word] for word in list(text)]

            # 判断长度，超过 SEQ_LEN的，则截取，不足的填充
            if len(indexes) >= config.SEQ_LEN:
                indexes = indexes[:config.SEQ_LEN]
            else:
                indexes += [config.PAD_INDEX] * (config.SEQ_LEN - len(indexes))

            return indexes
