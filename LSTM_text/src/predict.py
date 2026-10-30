#predict.py 预测数据
# 循环输入评论， 输出预测结果
import torch

from day0731.src import config
from day0731.src.model import ReviewAnalyzeModel
from day0731.src.tokenizer import JieBaTokenizer

if __name__ == '__main__':
    # 循环输入评论

    while True:

        review = input("请输入评论:")

        if review.strip() == "":
            print("内容为空，请重新输入")
        elif (review.lower().strip() == "exit" or
            review.lower().strip() == "q" ):
            break
        else:

            # 处理数据  创建分词器对象
            tokener = JieBaTokenizer.from_vocab(config.PROCESSED_DATA_DIR / "words.txt")

            # 分词并获得索引
            input_list = tokener.encode(review)

            # 转换成张量  注意：生成张量的时候，需要使用二维的列表
            input_list = torch.tensor([input_list],dtype=torch.long).to(config.DEVICE)

            # 调用模型进行预测
            model = ReviewAnalyzeModel(tokener.words_size)
            model.load_state_dict(torch.load(config.MODELS_DIR / "model.pth"))

            model.to(config.DEVICE)

            model.eval()
            with torch.no_grad():

                output = model(input_list)

                # 将结果使用sigmoid进行处理
                result = torch.sigmoid(output).tolist()

                print(result)

            # 将结果进行处理  如果大于等于0.5，结果是1，否则结果是0
            if result[0] > 0.5:
                print(f'正面评价（置信度：{result[0]:.2f}）')
            else:
                print(f'负面评价（置信度：{1-result[0]:.2f}）')







