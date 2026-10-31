# predict.py
# 动态输入中文，然后翻译成英文
import torch

from transformer import config
from transformer.model import TransformerModel
from transformer.tokenizer import MyTokenizer


def predict():

    # 输入内容
    while True:
        text = input("请输入中文(输入q或者exit退出)：")

        if text.lower() == "exit" or text.lower() == "q":
            break

        # 处理数据
        my_tokenizer = MyTokenizer(
            config.PROCESSED_PATH / "zh_vocab.txt",
            config.PROCESSED_PATH / "en_vocab.txt"
        )
        # 需要转换成二维  [1,64]
        input_list = [my_tokenizer.encode(text,False)]

        # 张量
        input_list = torch.tensor(input_list,dtype=torch.long).to(config.DEVICE)

        # 加载模型
        model = TransformerModel(
            len(my_tokenizer.zh_words),
            len(my_tokenizer.en_words),
        )
        model.load_state_dict(torch.load(config.MODELS_PATH/"model.pt"))

        model.to(config.DEVICE)
        # 开启预测评估模型
        model.eval()

        with torch.no_grad():
            # 编码器运行
            memory = model.encode(input_list)

            # 用于存放预测的结果
            result_list = []

            # 处理<sos>的数据
            en_input = torch.tensor([[config.SOS_INDEX]], dtype=torch.long).to(config.DEVICE)

            # 解码器运行  从<sos>开始，循环获取英文
            for i in range(config.SEQ_LEN):

                # 运行解码器
                output = model.decode(input_list,en_input,memory)

                # [1,i+1,7971]
                # 获取每次预测的最新的结果，概率最大的那个索引
                predict_index = output[:,-1,:].argmax(-1).item()

                # 如果预测的索引是 eos对应的，则直接退出
                if predict_index == config.EOS_INDEX:
                    break

                # 获取 预测的词结果
                result_list.append(my_tokenizer.en_words[predict_index])

                # 将预测的索引 加入到 en_input中
                en_input_list = en_input.tolist()[0]
                en_input_list.append(predict_index)

                en_input = torch.tensor([en_input_list],dtype=torch.long).to(config.DEVICE)

            # 转换成结果输出
            print("预测结果为:"," ".join(result_list))


if __name__ == '__main__':
    predict()
