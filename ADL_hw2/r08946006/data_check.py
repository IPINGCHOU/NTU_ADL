#%%
#checking tokenized answer span
#%%
import random
import numpy as np
import torch
import logging
import os
import json
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

TRAIN_ROUTE = './data/train.json'
VALID_ROUTE = './data/dev.json'
TEST_ROUTE = './data/test.json'

MAX_CONTEXT_LENGTH = 480
MAX_LENGTH = 512

#%%
# read json file
import json
with open(TRAIN_ROUTE, 'r') as reader:
    train = json.loads(reader.read())
with open(VALID_ROUTE, 'r') as reader:
    valid = json.loads(reader.read())

train = train['data']
valid = valid['data']

# %%
# get all paragraphs
train_paragraphs = []
valid_paragraphs = []

for data in train:
    train_paragraphs.append(data['paragraphs'])
for data in valid:
    valid_paragraphs.append(data['paragraphs'])

#%%
data_paragraphs = valid_paragraphs 
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

start = []
end = []
ans1 = 0
ans0 = 0
for paragraph in data_paragraphs:
    for sector in paragraph:
        context = sector['context']
        context = tokenizer.encode(context, add_special_tokens = False)
        for qa in sector['qas']:
            # prepare tokenized question + context
            question = tokenizer.encode(qa['question'], add_special_tokens = False)

            question_tokenized = tokenizer.batch_encode_plus([(context)])

            if qa['answerable']:
                ans1 += 1
                answer = tokenizer.encode(qa['answers'][0]['text'], add_special_tokens = False)
                match = find_sub_list(answer, question_tokenized['input_ids'][0])

                if len(match) == 0:
                    continue
                else:
                    start.append(match[0][0])
                    end.append(match[0][1])
            else:
                ans0 += 1

#%%
import matplotlib.pyplot as plt
plt.hist(start, bins = 40)
plt.show()
# %%
plt.hist(end, bins = 40)
plt.show()
#%%
diff = np.array(end) - np.array(start)
plt.hist(diff, bins = 40)
plt.show()

# %%
def preprocessing_dataset(data_paragraphs, flag, tokenizer):
    output_list = []

    #TODO flag

    ids = 0
    non_match_count = 0
    matched_count = 0
    for paragraph in data_paragraphs:
        print('Now ids: ' + str(ids), end = '\r')
        for sector in paragraph:
            context = sector['context']
            context = tokenizer.encode(context,
                                       add_special_tokens = False,
                                       max_length = MAX_CONTEXT_LENGTH,
                                       pad_to_max_length = True)
            for qa in sector['qas']:
                # prepare tokenized question + context
                question = tokenizer.encode(qa['question'], add_special_tokens = False)

                question_tokenized = tokenizer.batch_encode_plus([(context,question)],
                                                                max_length = MAX_LENGTH,
                                                                pad_to_max_length = True,
                                                                truncation_strategy='only_second')

                # prepare answerable, start, stop
                if qa['answerable'] == False:
                    answerable = 0
                    answer_start = -1
                    answer_end = -1
                else:
                    answerable = 1
                    answer = tokenizer.encode(qa['answers'][0]['text'], add_special_tokens = False)

                    match = find_sub_list(answer, question_tokenized['input_ids'][0][:MAX_CONTEXT_LENGTH+1])
                    if len(match) == 0:
                        non_match_count += 1
                        continue
                    else:
                        answer_start, answer_end = match[0]
                        matched_count += 1
                        answer_start -= 1
                        answer_end -= 1

                # prepare dict for question list and ans_list
                question_tokenized['id'] = ids
                question_tokenized['input_ids'] = question_tokenized['input_ids'][0]
                question_tokenized['token_type_ids'] = question_tokenized['token_type_ids'][0]
                question_tokenized['attention_mask'] = question_tokenized['attention_mask'][0]
                ids += 1

                ans_list = [answerable, answer_start, answer_end]
                question_tokenized['answer'] = ans_list
                output_list.append(question_tokenized)

    print('Non-matched sentences: ' + str(non_match_count))
    print('Matched sentences: ' + str(matched_count))
    print('==============')
    return output_list
            
#%%
# preprocessing datasets
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)

print('preprocessing training dataset...')
train_qa = preprocessing_dataset(train_paragraphs, 'train', tokenizer)
print('preprocessing valid dataset...')
valid_qa = preprocessing_dataset(valid_paragraphs, 'train', tokenizer)

# %%
# train_qa
start = []
end = []
for i,t in enumerate(train_qa):
    answer = t['answer']
    if answer[0]:
        start.append(answer[1])
        end.append(answer[2])

# %%
start = np.array(start)
end = np.array(end)
diff = end - start

import matplotlib.pyplot as plt
plt.hist(diff, bins = 40, density = True, cumulative = True, rwidth = 20)
plt.title('Cumulative Train set answer length after tokenized')
plt.ylabel('Count (%)')
plt.show()

# %%
import matplotlib.pyplot as plt
x = [0.1,0.3,0.5,0.7,0.9]
answerable_em = [0.78547, 0.78376, 0.78206, 0.77866, 0.76898]
answerable_f1 = [0.83685, 0.83424, 0.83206, 0.82809, 0.81888] 
overall_em = [0.80850, 0.82618, 0.83353, 0.83631, 0.83353]
overall_f1 = [0.84447, 0.86151, 0.86853, 0.87091, 0.86784]
unanswerable_em = [0.86225, 0.92516, 0.95364, 0.97086, 0.98211]

fig, axs = plt.subplots(1,2, sharey=True, figsize = (12,6))
plt.setp(axs, xticks = x)
axs[0].plot(x, answerable_f1, '-o', label = 'answerable')
axs[0].plot(x, overall_f1, '-o', label = 'overall')
axs[0].plot(x, unanswerable_em, '-o', label = 'unanswerable')

axs[1].plot(x, answerable_em, '-o')
axs[1].plot(x, overall_em, '-o')
axs[1].plot(x, unanswerable_em, '-o')

axs[0].set_title('F1')
axs[1].set_title('EM')
fig.legend(loc = 'upper right')
plt.show()








# %%
