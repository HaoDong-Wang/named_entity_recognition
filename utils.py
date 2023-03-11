import logging
import os
import time
import torch
from transformers import BertTokenizer
from datetime import timedelta



def get_Logger(config, dataset):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s|%(filename)s|%(levelname)s] %(message)s',
                                  datefmt='%a %b %d %H:%M:%S %Y')

    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    # work_dir = os.path.join(config.log_path, time.strftime("%Y-%m-%d-%H.%M", time.localtime()))
    work_dir = os.path.join(config.log_path, dataset.split('/')[-1], config.model_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='a')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger

def get_wordsid(VOC_dir):
    word2ids={}
    id2words={}
    with open(VOC_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line=='\n':
                break
            word2ids[line.split('\t')[0]]=int(line.split('\t')[1])
            id2words[int(line.split('\t')[1])]=line.split('\t')[0]
    return word2ids,id2words

class NERDataset(object):
    def __init__(self, f_path, max_len, word2ids, class2id, config, is_bert=False):
        self.sents = []
        self.tags_li = []
        self.class2id = class2id
        self.word2ids = word2ids
        self.is_bert = is_bert
        if self.is_bert:
            self.bert = BertTokenizer.from_pretrained(config.bert_path)

        with open(f_path, 'r', encoding = 'utf-8') as f:
            lines = [line.split('\n')[0] for line in f.readlines() if len(line.strip())!=0]

        tags = [line.split('\t')[1] for line in lines]
        words = [line.split('\t')[0] for line in lines]
        word, tag = [], []
        for char, t in zip(words, tags):
            if char != '。':
                word.append(char)
                tag.append(t)
            else:
                if len(word) > max_len:
                    self.sents.append(['[CLS]'] + word[:max_len] + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag[:max_len] + ['[SEP]'])
                else:
                    self.sents.append(['[CLS]'] + word + ['[SEP]'])
                    self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        if self.is_bert:
            sents, tags = self.sents[idx], self.tags_li[idx]
            token_ids = self.bert.encode(sents, add_special_tokens=False)
            label_ids = [self.class2id[x] for x in tags]
        else:
            sents, tags = self.sents[idx], self.tags_li[idx]
            token_ids = [self.word2ids[word] for word in sents]
            label_ids = [self.class2id[x] for x in tags]
            seqlen = len(label_ids)
        return token_ids, label_ids, seqlen

    def __len__(self):
        return len(self.tags_li)


def PadBatch(batch, config):
    if config.is_bert:
        bert = BertTokenizer.from_pretrained(config.bert_path)
        pad = bert.encode('[PAD]', add_special_tokens=False)[0]
    else:
        pad = config.word2ids['[PAD]']
    maxlen = config.pad_size
    token_tensors = torch.LongTensor([i[0] + [pad] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor([i[1] + [pad] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors != pad)
    return token_tensors, label_tensors, mask


class collater():
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        '''在这里重写collate_fn函数'''
        if self.config.is_bert:
            bert = BertTokenizer.from_pretrained(self.config.bert_path)
            pad = bert.encode('[PAD]', add_special_tokens=False)[0]
        else:
            pad = self.config.word2ids['[PAD]']
        maxlen = self.config.pad_size
        token_tensors = torch.LongTensor([i[0] + [pad] * (maxlen - len(i[0])) for i in batch])
        label_tensors = torch.LongTensor([i[1] + [pad] * (maxlen - len(i[1])) for i in batch])
        mask = (token_tensors != pad)
        return token_tensors, label_tensors, mask


def build_dataset(config):
    def load_dataset(path, pad_size):
        contents = []
        with open(path, 'r', encoding = 'utf-8') as f:
            words = []
            tags = []
            for line in f.readlines():
                lin = line.strip()
                word, tag = lin.split('\t')[0], lin.split('\t')[1]
                if word != '。':
                    words.append(word)
                    tags.append(tag)
                else:
                    token = ['[CLS]'] + words
                    tags = ['O'] + tags
                    seq_len = len(token)
                    if pad_size > len(tags):
                        mask = [1] * len(tags) + [0] * (pad_size - len(tags))
                        token += ['[PAD]'] * (pad_size - len(token))
                        tags += ['O'] * (pad_size - len(tags))
                    else:
                        mask = [1] * pad_size
                        token = token[:pad_size]
                        tags = tags[:pad_size]
                        seq_len = pad_size
                    token_ids = config.tokenizer.convert_tokens_to_ids(token)
                    tags_ids = [config.class2id[x] for x in tags]
                    contents.append((token_ids, tags_ids, seq_len, mask))
                    words = []
                    tags = []
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batches = batches
        self.batch_size = batch_size
        self.device = device
        self.n_batch = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数, False为整数
        if len(batches) % self.n_batch != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, batches):
        x = torch.LongTensor([_[0] for _ in batches]).to(self.device)
        y = torch.LongTensor([_[1] for _ in batches]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in batches]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in batches]).to(self.device)

        mask = (mask > 0)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batch:
            batches = self.batches[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            return self._to_tensor(batches)
        elif self.index >= self.n_batch:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batch + 1
        else:
            return self.n_batch

def build_iter(dataset, batch_size, device):
    return DatasetIterater(dataset, batch_size, device)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
