# model.py
# 创建模型 - 标准Transformer（Encoder + Decoder）
import math
import torch
from torch import nn

import config
from process import load_vocab


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 加载词表，确定词表大小
        vocab = load_vocab()
        self.vocab_size = len(vocab)
        self.pad_id = vocab[config.PAD_TOKEN]
        self.bos_id = vocab[config.BOS_TOKEN]

        # 词嵌入层（编码器和解码器共用）
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=config.EMBED_DIM,
            padding_idx=self.pad_id
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(config.EMBED_DIM)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.N_HEAD,
            dim_feedforward=config.FFN_DIM,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.NUM_LAYERS
        )

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.N_HEAD,
            dim_feedforward=config.FFN_DIM,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.NUM_LAYERS
        )

        # 输出层（10分类）
        self.fc_layer = nn.Linear(config.EMBED_DIM, 10)

    #前向传播
    def forward(self, input_list):

        # ============ 编码器部分 ============
        # 词向量 [batch, seq_len] -> [batch, seq_len, embed_dim]
        enc_embed = self.embedding(input_list)
        enc_embed = self.positional_encoding(enc_embed)

        # 生成key_padding_mask: True位置会被mask掉（pad_id的位置）
        src_key_padding_mask = (input_list == self.pad_id)

        # Transformer编码
        memory = self.encoder(
            enc_embed,
            src_key_padding_mask=src_key_padding_mask
        )
        # memory: [batch, seq_len, embed_dim]  编码器输出（作为解码器的KV）

        # ============ 解码器部分 ============
        # 解码器输入：一个[BOS]标记，通过交叉注意力关注编码器输出
        batch_size = input_list.size(0)
        tgt = torch.full(
            (batch_size, 1), self.bos_id,
            dtype=torch.long, device=input_list.device
        )  # [batch, 1]

        # 解码器输入的词向量 + 位置编码
        dec_embed = self.embedding(tgt)
        dec_embed = self.positional_encoding(dec_embed)

        # 解码器对memory进行交叉注意力
        dec_output = self.decoder(
            tgt=dec_embed,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask
        )
        # dec_output: [batch, 1, embed_dim]

        # 取解码器输出做分类
        pooled = dec_output[:, 0, :]  # [batch, embed_dim]

        # 输出logits
        return self.fc_layer(pooled)
