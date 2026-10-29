# evaluate.py
# 模型评估：在测试集上计算分类准确率

import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TransformerModel
from tokenizer import MyTokenizer


def evaluate():
    # 加载测试集
    test_loader = get_dataloader(config.PROCESSED_PATH/'test.jsonl')

    # 创建分词器，拿到词表大小
    my_tokenizer = MyTokenizer(config.PROCESSED_PATH/'vocab.txt')

    # 加载模型（并加载训练好的权重）
    model = TransformerModel(len(my_tokenizer.words), config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODELS_PATH/'model.pt'))
    model.to(config.DEVICE)
    model.eval()


    cnt = 0       # 预测正确的个数
    all_cnt = 0   # 总个数

    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc='预测'):
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.tolist()

            output = model(batch_x)  # (batch, num_classes)
            predict_list = output.argmax(-1).tolist()   # 取得分最大的类别

            for i in range(len(predict_list)):
                if predict_list[i] == batch_y[i]:
                    cnt += 1
            all_cnt += len(batch_y)

    print('======= 评估结果 =======')
    print(f'准确率: {cnt / all_cnt:.4f}')
    print('========================')


if __name__ == '__main__':
    evaluate()
