# model.py
# 创建模型
import math

import torch
from torch import nn

from transformer import config


class TransformerModel(nn.Module):

    def __init__(self,zh_size,en_size):
        super().__init__()

        # 编码器
        # 嵌入层
        self.en_emb_layer = nn.Embedding(zh_size,config.EDIM)

        # 解码器
        # 嵌入层
        self.de_emb_layer = nn.Embedding(en_size,config.EDIM)
        # 位置编码

        # transformer模型
        self.transformer = nn.Transformer(
            d_model=config.EDIM,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            batch_first=True
        )

        # 输出层
        self.de_fc_layer = nn.Linear(config.EDIM,en_size)

        # 位置编码
        pos_list = []
        for pos in range(config.SEQ_LEN):
            vec = []
            for i in range(config.EDIM):
                num = pos / 10000 ** (i // 2 * 2 / config.EDIM)
                # 如果i是偶数，sin
                if i % 2 == 0:
                    vec.append(math.sin(num))
                # 如果i是奇数，cos
                else:
                    vec.append(math.cos(num))
            pos_list.append(vec)

        self.pos_list = torch.tensor(pos_list, dtype=torch.float).to(config.DEVICE)

    # 前向传播
    def forward(self,input_list,target_list):
        # 调用编码器
        memory = self.encode(input_list)
        # 调用解码器
        res = self.decode(input_list,target_list,memory)

        return res

    # 编码器
    def encode(self, input_list):
        # 数据 嵌入层  位置编码
        src_emb = self.en_emb_layer(input_list)

        # 位置编码
        src_emb = src_emb + self.pos_list

        # 填充的掩码
        src_pad_mask = (input_list == config.PAD_INDEX)

        # 编码器运行
        memory = self.transformer.encoder(
            src = src_emb,
            src_key_padding_mask = src_pad_mask
        )

        return memory

    # 解码器
    def decode(self,input_list,target_list,memory):

        # target_list 是英文的输入 维度是SEQ_LEN-1
        # [32,63] target_list.shape[1]

        # 嵌入层
        tgt_emb = self.de_emb_layer(target_list)

        # 位置编码
        tgt_emb = tgt_emb + self.pos_list[:target_list.shape[1],:]

        # 掩码遮盖
        # target填充值
        tgt_pad_mask = (target_list == config.PAD_INDEX)

        # 交互 编码解码遮盖
        tgt_mask = (self.transformer.
                    generate_square_subsequent_mask(
                        target_list.shape[1]
                    )).to(config.DEVICE)

        # 输入的填充
        src_pad_mask = (input_list == config.PAD_INDEX)

        output = self.transformer.decoder(
            tgt = tgt_emb,
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_pad_mask,
            memory_key_padding_mask = src_pad_mask
        )

        # 输出层进行处理
        return self.de_fc_layer(output)
