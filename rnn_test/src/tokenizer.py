import jieba
from tqdm import tqdm


class JieBaTokenizer:

    def __init__(self,words):
        self.vocab_list = words
        self.vocab_size = len(words)

        # 处理词和索引的映射，并将其生成字典
        idx_to_word = {}
        word_to_idx = {}
        for i in range(len(words)):
            idx_to_word[i] = words[i]  # k=i,v=words[i]  i是索引
            word_to_idx[words[i]] = i  # k=words[i],v=i  i是索引

        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.unk_token_index = 0

    # 输入句子，进行分词，并返回分词结果list
    @staticmethod
    def tokenize(sentence):
        return jieba.lcut(sentence)

    # 输入所有的句子，进行分词，并将词表存入到文件中
    @classmethod
    def build_vocab(cls, sentences,file_path):
        # 转换成词表，并生成 词和索引互相转换的字典
        words = set()

        # 循环句子，并切分成单词，存入到words中。
        for s in tqdm(sentences, "进行分词"):
            words.update(jieba.lcut(s))

        # 将set集合转换成list列表，将<unk>放到第一位
        words = ["<unk>"] + list(words)

        print(len(words))

        # 可以将其存入到文件中
        with open(file_path, "w", encoding="utf-8") as f:
            for w in tqdm(words, "保存词到文件"):
                f.write(w + "\n")


    def encode(self, sentences):

        # 循环遍历sentences中每条语句，并分词，将词转换成对应的索引
        sent_idx = []  # 用于存放转换后的索引的list

        # 循环每一个句子
        for s in sentences:
            # 分词，得到结果是一个list列表，每个元素都是分词
            s_li = jieba.lcut(s)

            # 用于存放，转换后的所有的s_li中元素的索引
            s_idx = []

            for w in s_li:
                s_idx.append(self.word_to_idx.get(w, 0))

            # 将每个句子的索引列表，都放到sent_idx中
            sent_idx.append(s_idx)
        return sent_idx

    @staticmethod
    def get_words_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [ word.strip() for word in f.readlines()]


