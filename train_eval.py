import torch
import time
import numpy as np
from utils import get_time_dif
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup


def train(model, config, train_iter, dev_iter, test_iter, logger):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    # total_steps = train_iter.__len__()  * config.epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= config.warm_up_ratio * total_steps,
    #                                             num_training_steps=total_steps)
    losses = 0.0
    total_batch = 1     # 进行轮数
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    dev_best_loss = float('inf')

    for epoch in range(config.epochs):
        logger.info(f'=====第{epoch}轮训练=====')
        for i, batch in enumerate(train_iter):
            x, y, z = batch
            # x:input_ids; y:label_ids ;z:mask
            x = x.to(config.device)
            y = y.to(config.device)
            z = z.to(config.device)
            loss = model(x, y, z)
            losses+=loss.item()

            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            if total_batch % config.dev_batch == 0:
                p, r, f1, dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {:>6},  Val P: {:>5.4},  Val R: {:>6.4%},  Val F1: {:>5.4},  Val Acc: {:>6.4%},  Time: {} {}'
                logger.info(msg.format(total_batch, p, r, f1, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    # plt.show()


def evaluate(config, model, dev_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, batch in enumerate(dev_iter):
            x, y, z = batch
            x = x.to(config.device)
            y = y.to(config.device)
            z = z.to(config.device)
            predict = model(x, y, z, is_test = True)
            labels = y.data.cpu().numpy()
            for i in range(len(predict)):
                predict_all = np.append(predict_all, predict[i])
                labels_all = np.append(labels_all, labels[i][:len(predict[i])])

        p = metrics.precision_score(labels_all, predict_all, average='macro')
        r = metrics.recall_score(labels_all, predict_all, average='macro')
        f1 = metrics.f1_score(labels_all, predict_all, average='macro')
        acc = metrics.accuracy_score(labels_all, predict_all)
        return p, r, f1, acc, loss_total / len(dev_iter)

