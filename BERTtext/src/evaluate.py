# evaluate.py
# 模型评估
import torch
from tqdm import tqdm

from BERTtext.src import config
from BERTtext.src.dataset import get_dataloader
from BERTtext.src.model import BertModel


# 智能候选词         top1 top5
# 智能二分类         准确率  预测结果正确的个数/总个数
# 智能中英文翻译     bleu

def evaluate():
    # 加载数据  测试集
    dataloader = get_dataloader(config.PROCESSED_PATH / "test.jsonl")

    # 加载模型
    model = BertModel()
    model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))

    # 预测
    model.to(config.DEVICE)
    model.eval()

    # 准确的个数
    cnt = 0
    # 总个数
    all_cnt = 0

    for input_list,attention_list,target_list in tqdm(dataloader,"批量预测"):
        # 设置数据的设备
        input_list = input_list.to(config.DEVICE)
        attention_list = attention_list.to(config.DEVICE)
        target_list = target_list.tolist()

        # 获得预测结果
        output = model(input_list,attention_list)

        # 将结果进行sigmoid处理
        predict_list = torch.round(torch.sigmoid(output)).tolist()

        # target_list 真实值
        # predict_list 预测值
        for i in range(len(predict_list)):
            if predict_list[i] == target_list[i]:
                cnt += 1

        # 统计总个数
        all_cnt += len(target_list)

    # 评估
    print("准确率为：",(cnt/all_cnt))

if __name__ == '__main__':
    evaluate()

