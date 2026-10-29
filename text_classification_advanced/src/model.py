# model.py
# 手动搭建 Transformer 编码器模型（用于文本分类）

import math

import torch
from torch import nn

import config


class TransformerModel(nn.Module):

    def __init__(self, vocab_size, num_classes):
        super().__init__()

        # 词嵌入层
        self.emb_layer = nn.Embedding(vocab_size, config.EDIM)

        # Transformer 编码器：由 NUM_LAYERS 个编码层堆叠而成
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EDIM,
            nhead=config.NHEAD,
            batch_first=True, )   # 单层：d_model=EDIM, nhead=NHEAD, batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_LAYERS
        )   # 堆叠 num_layers 层

        # 分类输出层：整句表示 -> 各类别得分
        self.fc_layer = nn.Linear(config.EDIM, num_classes)

        # 位置编码（sin / cos）
        # 公式：num = pos / 10000 ** (2i / d_model)
        # TODO: 遍历每个位置 pos 和每个维度 i：
        #       偶数维度用 sin，奇数维度用 cos
        pos_list = []
        for pos in range(config.SEQ_LEN):
            vec = []
            for i in range(config.EDIM):
                num = pos/10000**(i//2*2/config.EDIM)
                if i%2==0:
                    vec.append(math.sin(num))
                else:
                    vec.append(math.cos(num))
            pos_list.append(vec)

        # 注册为 buffer：随模型一起移动设备，但不会作为参数被优化
        # 转为张量矩阵
        self.register_buffer('pos_emb', torch.tensor(pos_list,dtype=torch.float))

    def forward(self, input_list):
        # input_list.shape: (batch_size, seq_len)

        # 词嵌入 + 位置编码
        emb = self.emb_layer(input_list)   # (batch, seq, edim)
        emb = emb+self.pos_emb[:input_list.shape[1],:]

        # 填充掩码：填充位置不参与注意力计算
        pad_mask = (input_list == config.PAD_INDEX)   # (batch, seq)

        # 编码器前向
        output = self.transformer_encoder(emb,src_key_padding_mask = pad_mask,)   # (batch, seq, edim)

        # 取第一个位置的输出作为整句聚合表示（类似 BERT 的 [CLS]）
        cls_output = output[:,0,:]   # (batch, edim)

        # 分类层输出得分
        return self.fc_layer(cls_output)   # (batch, num_classes)
