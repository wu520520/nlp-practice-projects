# evaluate.py
# 评估模型
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from transformer import config
from transformer.dataset import get_dataloader
from transformer.model import TransformerModel
from transformer.tokenizer import MyTokenizer


def evaluate():

    # 加载数据
    test_dataloader = get_dataloader(config.PROCESSED_PATH / "test.jsonl")

    # 创建分词器对象
    my_tokenizer = MyTokenizer(
        config.PROCESSED_PATH / "zh_vocab.txt",
        config.PROCESSED_PATH / "en_vocab.txt"
    )

    # 加载模型
    model = TransformerModel(
        len(my_tokenizer.zh_words),
        len(my_tokenizer.en_words)
    )
    model.load_state_dict(torch.load(config.MODELS_PATH / "model.pt"))

    # 预测
    model.to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        all_pred_list = [] # 所有的预测结果
        all_ref_list = [] # 所有的真实结果

        for batch_x,batch_y in tqdm(test_dataloader,"批次预测"):
            batch_x = batch_x.to(config.DEVICE)
            batch_y = batch_y.tolist()

            for i in range(len(batch_y)):
                # [1,64]
                input_list = batch_x[i].unsqueeze(0)

                # 用于存放预测的词结果
                pre_list = []

                # 编码器运行
                memory = model.encode(input_list)

                # 解码器运行
                # sos
                en_input = torch.tensor([[config.SOS_INDEX]],dtype=torch.long).to(config.DEVICE)

                # 一个词一个词的处理
                for j in range(config.SEQ_LEN):

                    # 解码器运行
                    output = model.decode(input_list,en_input,memory)

                    # 获得预测的索引值
                    predict_index = output[:,-1,:].argmax(-1).item()

                    if predict_index == config.EOS_INDEX:
                        break

                    # 获取对应的词，并保存
                    pre_list.append(my_tokenizer.en_i2w[predict_index])

                    # 更新 en_input的值
                    en_input_list = en_input.tolist()[0]
                    en_input_list.append(predict_index)

                    en_input = torch.tensor([en_input_list],dtype=torch.long).to(config.DEVICE)

                # 将预测结果存入到 all_pred_list中
                all_pred_list.append(pre_list)

                # 将真实结果进行处理，也存入到all_ref_list中
                # 去掉填充 sos eos，然后将索引转换成词
                ref_list = []

                for idx in batch_y[i]:
                    if idx != config.PAD_INDEX and idx != config.SOS_INDEX and idx != config.EOS_INDEX:
                        ref_list.append(my_tokenizer.en_i2w[idx])

                all_ref_list.append([ref_list])
    # 评估
    print("bleu的值为：",corpus_bleu(all_ref_list, all_pred_list))


if __name__ == '__main__':
    evaluate()