import torch
import numpy as np
from utils import NERDataset, get_wordsid,PadBatch
import argparse
from torch.utils import data
from importlib import import_module

dataset = './datasets/CMeEE'  # 数据集, 注：末尾不带 /
parser = argparse.ArgumentParser(description='Chinese Named Entity Recognition')
parser.add_argument('--model', type=str, default='bilstm', required=False, help='choose a bilstm')
args = parser.parse_args()
model_name = args.model
x = import_module('models.' + model_name)
config = x.Config(dataset)

# 数据集
dataset = 'datasets/CMeEE'
word2ids, id2words = get_wordsid(dataset + '/data/vocab.txt')
test_dataset = NERDataset(config.test_path, config.pad_size - 2, word2ids, config.class2id, config)

test_iter = data.DataLoader(dataset=test_dataset,
                            batch_size=(config.batch_size) // 2,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=PadBatch)

# 加载模型
model = x.Model(config)
if __name__ == '__main__':
    model.load_state_dict(torch.load((config.save_path)))
    model.eval()
    predict_all = []
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            x, y, z = batch
            x = x.to(config.device)
            y = y.to(config.device)
            z = z.to(config.device)
            predict = model(x, y, z, is_test = True)
            for i in range(len(predict)):
                predict_all.append(predict[i])
    all = []
    with open(dataset+'/data/test.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()

    # exit()
    # 获取所有句子
    sentences = []
    sentence = []
    for x in texts:
        if x.split('\t')[0]=='。':
            sentences.append(sentence+['。'])
            sentence = []
        else:
            sentence.append(x.split('\t')[0])
    assert len(sentences)==len(predict_all), '''句子数量与标签数量不符'''
    # 获取所有labels
    labels = []
    label = []
    for i in range(len(predict_all)):
        label = [config.class2id['O']] * len(sentences[i])
        # assert len(label) == len(predict_all[i]) or len(predict_all[i]) == config.pad_size-2, '''句子长度与标签数量不符'''
        for ii in range(len(predict_all[i])):
            label[ii] = predict_all[i][ii]
        labels.append(label)


    with open(config.result_path, 'w', encoding='utf-8') as ff:
        for i in range(len(sentences)):
            for x,y in zip(sentences[i], labels[i]):
                ff.write(x+'\t'+config.id2class[y]+'\n')


