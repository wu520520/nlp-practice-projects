# 预测数据
import torch

from day0729.rnn_test.src import config
from day0729.rnn_test.src.model import InputMethodModel
from day0729.rnn_test.src.tokenizer import JieBaTokenizer


# 用于处理数据的方法
def get_index(str):
    # 获取词文件对象
    words = JieBaTokenizer.get_words_from_file(config.PROCESSED_DATA_DIR/"words.txt")
    # 创建一个分词器对象
    tokener = JieBaTokenizer(words)

    # 得到了一个索引的list
    return  torch.tensor(tokener.encode(str), dtype=torch.long)

# 用于加载模型并预测的方法
def predict(index_list):
    words = JieBaTokenizer.get_words_from_file(config.PROCESSED_DATA_DIR / "words.txt")

    # 获取模型
    model = InputMethodModel(len(words))
    model.load_state_dict(torch.load(config.MODELS_DIR / "model.pth"))

    # 将模型和数据都转入到gpu中进行计算
    model = model.to(config.DEVICE)
    index_list  = index_list.to(config.DEVICE)

    # 预测数据
    model.eval()
    with torch.no_grad():

        # 调用模型，得到结果
        result = model(index_list)

        # 根据结果，获取前k个值
        res_index_li =  torch.topk(result,k=5).indices.cpu().tolist()

        res_words = []
        for idx in res_index_li[-1]:
            res_words.append(words[idx])

        return  res_words

# main方法
if __name__ == '__main__':
    str_history = "" # 用于记录历史字符串

    while True:
        instr = input("请输入下一个词语：")

        # 退出条件
        if instr.lower() == "exit" or instr.lower() == "q":
            break

        # 拼接最新的字符串
        str_history += instr

        index_li = get_index(str_history)
        print(index_li)

        # 预测数据
        res_list = predict(index_li)

        print("历史输入：" + str_history)
        print("预测：" + str(res_list))





