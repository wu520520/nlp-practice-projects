import jieba
from tqdm import tqdm

from day0731.src import config


class JieBaTokenizer():

    def __init__(self,words_list):
        self.words = words_list
        self.words_size = len(words_list)

        # 设置 词和索引的映射字典
        index_to_word = {}
        word_to_index = {}

        for i in range(len(words_list)):
            index_to_word[i] = words_list[i]
            word_to_index[words_list[i]] = i

        # 赋值到实例变量中
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index

    # 构建词表
    @staticmethod
    def build_vocab(reviews_list,file_path):
        words = set()
        # 循环遍历，进行分词
        for review in tqdm(reviews_list,"分词"):
            words.update(jieba.lcut(review))

        # "<unk>" 匹配未知的词,"<pad>" 用于填充词
        words = ["<unk>","<pad>"] + list(words)

        # 保存到文件中
        with open(file_path,"w",encoding="utf-8") as f:
            for word in words:
                f.write(word + "\n")

    # 加载词表
    @staticmethod
    def from_vocab(file_path):
        words = []
        # 加载词表文件数据
        with open(file_path,"r",encoding="utf-8") as f:
            words = [line.strip("\n") for line in f.readlines()]

        # 创建一个对象并返回
        return JieBaTokenizer(words)


    # 将一个句子进行分词，并转换成索引list
    def encode(self,review):
        # 先分词
        review_words = jieba.lcut(review)

        # 用于存放分词后的索引列表
        review_indexes = []
        # 如果词的个数，超过了128SEQ_LEN
        if len(review_words) > config.SEQ_LEN:
            # 只截取前128个元素，并获得其索引，存入到review_indexes
            for i in range(config.SEQ_LEN):
                review_indexes.append(self.word_to_index[review_words[i]])
        else:
            # 不过不够128，则将所有的元素都转换成索引，存入到review_indexes
            for i in range(len(review_words)):
                review_indexes.append(self.word_to_index[review_words[i]])
            # 补全数据也存入到review_indexes中
            review_indexes += [config.PAD_INDEX] * (config.SEQ_LEN - len(review_words))

        return review_indexes


