# from transformers import BertTokenizer
#
# bert = BertTokenizer.from_pretrained('bert-base-chinese')
# print(bert.encode('我们是朋友', add_special_tokens=False))
# print(bert.encode('[PAD]', add_special_tokens=False))
#
# print(bert.convert_ids_to_tokens(bert(['大家好[PAD]'])['input_ids'][0]))

# mask = [True, False]
# print(sum(mask))

label = [0]*10
print(label)