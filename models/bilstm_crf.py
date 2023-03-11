import os
import torch
import torch.nn as nn
from TorchCRF import CRF
from utils import get_wordsid

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bilstm_crf'
        self.dataset = dataset
        self.log_path = './logs/'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.result_path = dataset + '/result/result.txt'
        os.makedirs(dataset + '/result/' , exist_ok = True)
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        os.makedirs(dataset + '/saved_dict/' , exist_ok = True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.corpus_num = len(open(dataset + '/data/vocab.txt', 'r', encoding='utf-8').readlines())+1
        self.class_list = [x.strip() for x in open(
            dataset + '/data/classes.txt').readlines()]                                 # 类别名单
        self.id2class = {}
        self.class2id = {}
        for id, c in enumerate(self.class_list):
            self.id2class[id] = c
            self.class2id[c] = id
        self.num_classes = len(self.class_list)                         # 类别数
        self.word2ids, self.id2words = get_wordsid(dataset+'/data/vocab.txt')

        self.is_bert = False
        self.epochs = 50                                                 # 训练轮数
        self.dev_batch = 300                                            #验证轮数
        self.require_improvement = 100000000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.batch_size = 24                                           # mini-batch大小
        self.pad_size = 124                                           # 句子长度大小
        self.learning_rate = 5e-5                                       # 学习率
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.droprate = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.corpus_num,config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size = config.embedding_dim,
            hidden_size = config.hidden_dim//2,
            num_layers = 2,
            bidirectional = True,
            batch_first = True)

        self.dropout = nn.Dropout(config.droprate)
        self.linear = nn.Linear(config.hidden_dim, config.num_classes)
        self.crf = CRF(config.num_classes)

    def _get_features(self, sentence, mask):
        encode_out=self.embedding(sentence)
        # encode_out, pooled=self.bert(sentence, attention_mask=mask, output_all_encoded_layers=False)
        enc, _ = self.lstm(encode_out)
        feats = self.linear(enc)
        return feats

    def forward(self, x, tags, mask, is_test=False):
        emissions = self._get_features(x, tags)
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(emissions, tags, mask).mean()
            return loss
        else:
            decode = self.crf.viterbi_decode(emissions, mask)
            return decode


