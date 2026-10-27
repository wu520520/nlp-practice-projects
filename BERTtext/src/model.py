# model.py
# 创建模型
from torch import nn
from transformers import AutoModel

from BERTtext.src import config


class BertModel(nn.Module):

    def __init__(self,is_freeze_bert=True):
        super().__init__()

        # 加载模型
        self.model = AutoModel.from_pretrained(
            config.PRETRAINED_PATH
        )

        # 不更新bert模型的参数
        if is_freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

        # 输出层
        self.fc_layer = nn.Linear(self.model.config.hidden_size,1)

    #前向传播
    def forward(self, input_list,attention_list):

        # bert的模型如何训练
        output = self.model(input_ids=input_list, attention_mask=attention_list)

        # [cls]对应的结果
        output = output.last_hidden_state[:, 0, :]

        #[[1],[0],[1]] => [1,0,1]
        return self.fc_layer(output).squeeze(-1)

