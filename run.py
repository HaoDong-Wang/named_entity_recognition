import argparse
import numpy as np
import torch
from torch.utils import data
from utils import get_Logger, NERDataset, PadBatch, collater
from utils import build_dataset, DatasetIterater
from importlib import import_module
from train_eval import train


dataset = './datasets/CMeEE'  # 数据集, 注：末尾不带 /
parser = argparse.ArgumentParser(description='Chinese Named Entity Recognition')
parser.add_argument('--model', type=str, default='bilstm_crf', required=False, help='choose a bilstm')

args = parser.parse_args()
if __name__ == '__main__':

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    logger = get_Logger(config, dataset)

    np.random.seed(123)
    torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

########### 1. 创建数据集法1
    # word2ids, id2words = get_wordsid(dataset+'/data/vocab.txt')
    train_dataset = NERDataset(config.train_path, config.pad_size-2, config.word2ids, config.class2id, config)
    eval_dataset = NERDataset(config.dev_path, config.pad_size-2, config.word2ids, config.class2id, config)
    test_dataset = NERDataset(config.test_path, config.pad_size-2,  config.word2ids, config.class2id, config)

    collate_fn = collater(config)
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=collate_fn)

    dev_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=(config.batch_size) // 2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=collate_fn)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(config.batch_size) // 2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=collate_fn)
    ########### 2. 创建数据集法2
    # train_data, dev_data, test_data = build_dataset(config)
    # train_iter = DatasetIterater(train_data, config.batch_size, config.device)
    # dev_iter = DatasetIterater(dev_data, config.batch_size, config.device)
    # test_iter = DatasetIterater(test_data, config.batch_size, config.device)

    logger.info('数据加载完成')

    model = x.Model(config).to(config.device)
    logger.info('模型加载完成')

    logger.info('开始训练')
    train(model, config, train_iter, dev_iter, test_iter, logger)
