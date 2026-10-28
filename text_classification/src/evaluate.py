# evaluate.py
# 评估模块：使用测试集对训练好的模型进行评估，输出准确率
import torch
from tqdm import tqdm
from text_classification.src import config
from text_classification.src.dataset import get_dataloader
from text_classification.src.model import BertModel


def evaluate():
    # 1. 加载测试集数据
    dataloader = get_dataloader(config.PROCESSED_PATH / "test.jsonl")

    # 2. 加载训练好的模型
    model = BertModel()
    model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))

    # 3. 切换到评估模式
    model.to(config.DEVICE)
    model.eval()

    cnt = 0       # 预测正确的样本数
    all_cnt = 0   # 总样本数

    # 4. 不计算梯度，进行批量预测
    with torch.no_grad():
        for input_list, attention_list, target_list in tqdm(dataloader, "批量预测"):
            input_list = input_list.to(config.DEVICE)
            attention_list = attention_list.to(config.DEVICE)

            # 模型输出 logits
            output = model(input_list, attention_list)

            # 取 logits 最大的类别作为预测结果
            predict_list = torch.argmax(output, dim=-1).tolist()
            target_list = target_list.tolist()

            # 统计预测正确的数量
            for i in range(len(predict_list)):
                if predict_list[i] == target_list[i]:
                    cnt += 1

            all_cnt += len(target_list)

    # 5. 输出准确率
    print("准确率为：", cnt / all_cnt)


if __name__ == '__main__':
    evaluate()
