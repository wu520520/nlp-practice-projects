# model.py
# 创建模型
from torch import nn

from day0731.src import config


class ReviewAnalyzeModel(nn.Module):

    # 初始化
    def __init__(self,words_size):
        super().__init__()

        # 嵌入层
        self.emb_layer = nn.Embedding(
            words_size,
            config.EMBEDDING_DIM,
            config.PAD_INDEX)

        # lstm层
        self.lstm_layer = nn.LSTM(
            config.EMBEDDING_DIM,
            config.HIDDEN_DIM,
            batch_first=True
        )

        # 输出层
        self.fc_layer = nn.Linear(
            config.HIDDEN_DIM,
            1)


    # 前向传播
    def forward(self,x):

        # 对每一层都进行训练
        emb_out = self.emb_layer(x)

        lstm_out, _ = self.lstm_layer(emb_out)

        lstm_out = lstm_out[:,-1,:]
        result = self.fc_layer(lstm_out).squeeze(dim=1)

        return result

