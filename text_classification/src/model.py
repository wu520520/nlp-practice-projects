# model.py
# 模型定义模块：基于 BERT 的多分类模型
from torch import nn
from transformers import AutoModel

from text_classification.src import config


class BertModel(nn.Module):
    """
    BERT 多分类模型
    在预训练 BERT 基础上，添加一个全连接输出层，输出 NUM_CLASSES 个类别的 logits
    全参数微调：BERT 所有参数均可训练
    """

    def __init__(self):
        super().__init__()

        # 加载预训练 BERT 模型（bert-base-chinese）
        self.model = AutoModel.from_pretrained(
            config.PRETRAINED_PATH
        )

        # 多分类输出层：BERT 隐藏层维度 → 类别数（10） 768➡️10
        self.fc_layer = nn.Linear(self.model.config.hidden_size, config.NUM_CLASSES)

    def forward(self, input_list, attention_list):
        """
        前向传播
        :param input_list: 输入 token ids [batch_size, seq_len]
        :param attention_list: 注意力掩码 [batch_size, seq_len]
        :return: 各类别的 logits [batch_size, NUM_CLASSES]
        """
        # BERT 前向计算，得到最后一层隐藏状态
        output = self.model(input_ids=input_list, attention_mask=attention_list)

        # 取 [CLS] token 的输出（每个样本序列的第 0 个位置）作为整句的表示
        output = output.last_hidden_state[:, 0, :]

        # 通过全连接层输出各类别的 logits（未归一化的分数）
        return self.fc_layer(output)
