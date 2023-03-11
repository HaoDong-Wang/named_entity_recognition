import json

types = []
voc = {}
# 保存数据
def saved_data(path, saved_path, is_test = False):
    words = []
    labels = []
    with open(path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
        for item in data:
            if is_test == False:
                text = item['text']
                word = list(text)
                label = ['O'] * len(word)
                entities = item['entities']
                for entitie in entities:
                    if entitie['type'] not in types:
                        types.append(entitie['type'])
                    label[int(entitie['start_idx'])] = 'B-' + entitie['type']
                    for i in range(int(entitie['start_idx'])+1, int(entitie['end_idx'])+1):
                        label[i] = 'I-' + entitie['type']
            else:
                text = item['text']
                word = list(text)
                label = 'O' * len(word)
            words += word
            labels += label

    with open(saved_path, 'w', encoding = 'utf-8') as f:
        for x, y in zip(words, labels):
            if x not in voc.keys():
                voc[x] = 0
            else:
                voc[x] += 1
            f.write(x + '\t' + y + '\n')



def saved_class(saved_path):
    with open(saved_path, 'w', encoding = 'utf-8') as f:
        # f.write('O\n[PAD]\n[SEP]\n[CLS]\n')
        f.write('O\n')
        for x in types:
            f.write('B-' + x + '\n')
            f.write('I-' + x + '\n')

def saved_voc(path, is_bert):
    with open(path, 'w', encoding='utf-8') as f:
        if is_bert:
            f.write('[CLS]\n[PAD]\n')
            for key in voc.keys():
                f.write(key + '\n')
        else:
            f.write('[CLS]\t0\n[PAD]\t1\n[SEP]\t2\n')
            i = 3
            for key in voc.keys():
                f.write(key + '\t' + str(i) + '\n')
                i+=1
saved_data('./CMeEE_train.json', './data/train.txt', is_test = False)
saved_data('./CMeEE_dev.json', './data/dev.txt', is_test = False)
saved_data('./CMeEE_test.json', './data/test.txt', is_test = True)

saved_class('./data/classes.txt')
# saved_voc('../../bert-pretrain/vocab.txt', is_bert = True)
saved_voc('./data/vocab.txt', is_bert = False)

