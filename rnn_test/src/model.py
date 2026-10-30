from torch import nn

from day0729.rnn_test.src import config


class InputMethodModel(nn.Module):

    def __init__(self, vocab_size):
        # 模型的初始化
        super().__init__()

        # 嵌入层  词的个数，词向量的维度
        self.emb_layer = nn.Embedding(vocab_size, config.EMBEDDING_DIM)

        # rnn
        self.rnn_layer = nn.RNN(config.EMBEDDING_DIM, config.HIDDEN_DIM, batch_first=True)

        # 输出层
        self.fc_layer = nn.Linear(config.HIDDEN_DIM, vocab_size)

    # 向前传播
    def forward(self, x):

        # 前向传播
        res1 = self.emb_layer(x)

        res2, _ = self.rnn_layer(res1)

        # 获取最后的隐藏状态
        res3 = res2[:, -1, :]

        return self.fc_layer(res3)


