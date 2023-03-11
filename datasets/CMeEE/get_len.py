import json

path = './CMeEE_train.json'
lens = {}
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        text = item['text']
        l = len(text)
        if l in lens.keys():
            lens[l]+=1
        else:
            lens[l]=0
print(lens)

import numpy as np

import matplotlib.pyplot as plt

for key in lens.keys():
    plt.scatter(key, lens[key])
plt.xlim(0, 200)
plt.ylim(0,400)
plt.show()

