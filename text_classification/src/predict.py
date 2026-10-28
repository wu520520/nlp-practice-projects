# predict.py
# 预测模块：输入新闻标题，输出预测的分类结果
import json
import torch
from transformers import AutoTokenizer

from text_classification.src import config
from text_classification.src.model import BertModel

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_PATH)

# 加载标签映射字典（id → 标签），用于将预测的类别 id 转换回中文标签
with open(config.PROCESSED_PATH / "label_map.json", 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# 将 json 中的字符串 key 转换为 int
id2label = {int(k): v for k, v in label_map["id2label"].items()}

# 加载训练好的模型
model = BertModel()
model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))
model.to(config.DEVICE)
model.eval()  # 切换到评估模式，关闭 dropout

# 交互式循环：用户输入标题，模型输出预测分类
while True:
    text = input("请输入新闻标题：")

    # 输入 exit 或 q 退出
    if text.lower() in ("exit", "q"):
        break

    # 对输入文本进行分词（return_tensors="pt" 返回 PyTorch 张量）
    text_list = tokenizer(
        text,
        padding="max_length",
        max_length=config.SEQ_LEN,
        truncation="longest_first",
        return_tensors="pt"
    )

    input_ids = text_list["input_ids"].to(config.DEVICE)
    attention_mask = text_list["attention_mask"].to(config.DEVICE)

    # 不计算梯度，加速推理
    with torch.no_grad():
        # 模型输出为各类别的 logits
        output = model(input_ids, attention_mask)

        # 使用 softmax 将 logits 转换为概率分布
        probs = torch.softmax(output, dim=-1)

        # 取概率最大的类别作为预测结果（与二分类的 sigmoid + round 不同）
        pred_id = torch.argmax(probs, dim=-1).item()

        # 该类别的置信度（概率值）
        confidence = probs[0][pred_id].item()

        # 输出预测分类和置信度
        print(f"预测分类: {id2label[pred_id]} (置信度: {confidence:.4f})")
