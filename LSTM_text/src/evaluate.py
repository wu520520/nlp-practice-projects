# evaluate.py 评估
# 根据测试集，使用模型进行预测，然后获取准确率
import torch

from day0731.src import config
from day0731.src.dataset import ReviewDataset
from day0731.src.model import ReviewAnalyzeModel
from day0731.src.tokenizer import JieBaTokenizer

if __name__ == '__main__':
    # 加载数据
    loader = ReviewDataset.get_dataloader(config.PROCESSED_DATA_DIR/"test.jsonl")

    # 创建模型
    tokenizer = JieBaTokenizer.from_vocab(config.PROCESSED_DATA_DIR/"words.txt")

    model = ReviewAnalyzeModel(tokenizer.words_size)
    model.load_state_dict(torch.load(config.MODELS_DIR/"model.pth"))

    cnt = 0
    total_cnt = 0
    # 训练
    model.to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:

            batch_x = batch_x.to(config.DEVICE)
            batch_y = list(batch_y)

            # 训练
            output = model(batch_x)

            # 将output中的数据，

            # 使用sigmoid转换成0 - 1的数据，
            # 然后跟0.5进行比较，大于0.5的为1，小于等于0.5的为0
            pred_list = []
            for i in range(len(output)):
                if torch.sigmoid(output[i]) > 0.5:
                    pred_list.append(1)
                else:
                    pred_list.append(0)

            # 判断pred_list 和batch_y 是否能对应上
            for i in range(len(batch_y)):
                if pred_list[i] == batch_y[i]:
                    cnt += 1

            total_cnt += len(batch_y)

    # 计算准确率
    print("准确率为:",cnt/total_cnt)




