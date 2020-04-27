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

DEVICE = 'cuda:0'

MODEL_SAVEPATH = 'model_context_truncated_447_max_480/'
CHOSED_BATCH = 1

MAX_LENGTH = 480
DROPOUT_RATE = 0.5
BATCH_SIZE =  6
BERT_LEARNING_RATE = 1e-05
LINEAR_LEARNING_RATE = 1e-05
EPOCH = 20


THRESHOLD = 0.4
ANS_CUT = 30
MAX_CONTEXT_LENGTH = 447

print('testing dev data')
TEST_INPUT = './data/dev.json'

#%%
with open(TEST_INPUT, 'r') as reader:
    test = json.loads(reader.read())

test = test['data']
test_paragraphs = []
for data in test:
    test_paragraphs.append(data['paragraphs'])

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

def preprocessing_dataset(data_paragraphs, flag, tokenizer):
    output_list = []

    #TODO flag

    non_match_count = 0
    matched_count = 0
    for ids, paragraph in enumerate(data_paragraphs):
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


                # prepare dict for question list and ans_list
                question_tokenized['id'] = qa['id']
                question_tokenized['input_ids'] = question_tokenized['input_ids'][0]
                question_tokenized['token_type_ids'] = question_tokenized['token_type_ids'][0]
                question_tokenized['attention_mask'] = question_tokenized['attention_mask'][0]

                output_list.append(question_tokenized)

    return output_list

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case = True)
print('preprocessing testing dataset')
test_qa = preprocessing_dataset(test_paragraphs, 'test', tokenizer)

from torch.utils.data import Dataset

class QA_dataset(Dataset):
    def __init__(self, data, flag):
        self.data = data
        self.flag = flag
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        # batch_qc : batch question + context

        batch_qc = []
        batch_segments = []
        batch_masks = []
        batch_answers = []
        batch_id = []

        if self.flag == 'train':
            for data in datas:
                batch_qc.append(data['input_ids'])
                batch_segments.append(data['token_type_ids'])
                batch_masks.append(data['attention_mask'])
                batch_answers.append(data['answer'])

            return torch.LongTensor(batch_qc), torch.LongTensor(batch_segments), torch.LongTensor(batch_masks), torch.LongTensor(batch_answers)
        else:
            for data in datas:
                batch_id.append(data['id']) # id type = string
                batch_qc.append(data['input_ids'])
                batch_segments.append(data['token_type_ids'])
                batch_masks.append(data['attention_mask'])
    
            return torch.LongTensor(batch_qc), torch.LongTensor(batch_segments), torch.LongTensor(batch_masks), batch_id

test_QA = QA_dataset(test_qa, 'test')

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class QA_Model(nn.Module):
    def __init__(self):
        super(QA_Model, self).__init__()

        self.hidden_dim = 768
        self.vocab_size = 21128 # should not be used, just in case

        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.answerable_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, 384),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(384, 192),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(96, 1)
        )

        self.start_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, 384),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(384, 192),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(96, 1)
        )

        self.end_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, 384),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(384, 192),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(192, 96),
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(96, 1)
        )

    def forward(self, qc, segment, mask):

        # sent BERT
        hidden_layer, _ = self.bert(input_ids = qc,
                                    attention_mask = mask,
                                    token_type_ids = segment)
        # sent answerable start end
        answerable = self.answerable_layer(hidden_layer[:,0])
        ans_start = self.start_layer(hidden_layer[:,1:MAX_CONTEXT_LENGTH+1])
        ans_end = self.end_layer(hidden_layer[:,1:MAX_CONTEXT_LENGTH+1])

        return answerable, ans_start, ans_end


from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader

model = QA_Model()
model.load_state_dict(torch.load(MODEL_SAVEPATH + 'model.epoch.{}'.format(CHOSED_BATCH)))
model.to(DEVICE)
model.eval()
dataloader = DataLoader(dataset=test_QA,
                        batch_size = BATCH_SIZE,
                        shuffle=False,
                        collate_fn=test_QA.collate_fn,
                        num_workers=4)
answerable_pred = []
answer_start_pred = []
answer_end_pred = []
answer_ids = []

trange = tqdm(enumerate(dataloader), total=len(dataloader), desc = 'predict')

for i, (qcs, segments, masks, ids) in trange:
    model.eval()
    qcs = qcs.to(DEVICE)
    segments = segments.to(DEVICE)
    masks = masks.to(DEVICE)

    answerable, ans_start, ans_end = model(qcs, segments, masks)
    # answerable adjust
    answerable = torch.sigmoid(answerable)
    answerable = answerable > THRESHOLD
    answerable_pred.extend(answerable.to('cpu'))
    # ans_start, end adjust

    _, start = ans_start.data.topk(1, dim = 1)
    _, end = ans_end.data.topk(1, dim = 1)
    start += 1
    end += 1

    answer_start_pred.extend(start.to('cpu').squeeze(-1))
    answer_end_pred.extend(end.to('cpu').squeeze(-1))

    # for j, (start, end) in enumerate(zip(ans_start, ans_end)):
    #     _, start = start.data.topk(1, dim = 0) 
    #     _, end = end.data.topk(1, dim = 0) 
    #     start += 1
    #     end += 1
    #     answer_start_pred.extend(start.to('cpu').squeeze(-1))
    #     answer_end_pred.extend(end.to('cpu').squeeze(-1))
    
    # ids
    answer_ids.extend(ids)

answerable_pred = torch.cat(answerable_pred).detach().numpy().astype(int)
answer_start_pred = torch.cat(answer_start_pred).detach().numpy().astype(int)
answer_end_pred = torch.cat(answer_end_pred).detach().numpy().astype(int)

#%%
QA_pred = {}
pred_ans1 = 0
pred_ans0 = 0
pred_ans0_0string = 0
bug_list1 = []
bug_list = []
bug_list2 = []

for sent_id, (now_id, answerable, ans_st, ans_end) in enumerate(zip(answer_ids, answerable_pred, answer_start_pred, answer_end_pred)):
    print('Now ids:' + str(sent_id), end='\r')

    ans_end += 1
    if answerable == 0:
        QA_pred[now_id] = ''
        pred_ans0 += 1
    else:
        # for exchanging
        if ans_st < ans_end:
            ans = test_qa[sent_id]['input_ids'][ans_st:ans_end]
        elif ans_st > ans_end:
            bug_list2.append((ans_st, ans_end))
            ans = test_qa[sent_id]['input_ids'][ans_end:ans_st]
        
        # non changing
        ans = test_qa[sent_id]['input_ids'][ans_st:ans_end]
        
        ans = np.array(ans)
        ans = ans[ans != 100]
        ans = ans.tolist()

        # ans = ans[:ANS_CUT]
        # discard too long answer
        if len(ans) > 37:
            QA_pred[now_id] = ''
            continue
        else:
            ans_string = tokenizer.decode(ans).replace(" ", "")
            QA_pred[now_id] = ans_string

        # print(ans_string)

        # ans_string = ans_string[:30]

#%%
print('len > 0: ' +str(pred_ans1) )
print('answerable false: ' + str(pred_ans0))
print('len = 0: ' + str(pred_ans0_0string))
print(bug_list)
print(bug_list1)
print(bug_list2)
 
# %%
# saving data

with open('context_trun_19.json', 'w') as fp:
    json.dump(QA_pred, fp)

#%%

# trun 450
# preprocess1 : if st > ed : x[ed:st], cut 38, threshold 0.5, model truncated epoch1
# preprocess2 : if st > ed : x[ed:st], cut 38, threshold 0.4, model truncated epoch 1
# preprcoess3 : same 1, threshold 0.3
# preprocess4 : same 1, threshold 0.2, cut 30
# preprocess5 : same 1, threshold 0.1, cut 25
# preprocess6 : same 1, threshold 0.1, cut 20

# trun 470
# preprocess7 : same 1, threshold 0.1, cut 30
# preprocess8 : no st>ed change, threshold 0.1, cut 30 

# trun 470 + freeze bert embedding, encoder layer 1
# preprocess9 : same1, threshold 0.1, cut 30
# preprocess10 : no st>ed change, threshold0.1, cut 30

# trun 430 total 480
# preprocess11: no st>ed, threshold 0.1 ,cut 30
# preprocess12: no st>ed, threshold 0.6, cut 30 
# preprocess13: exchange, threshold 0.5, cut 30
# preprocess14: exchange, threshold 0.4, no cut
# preprocess16: exchange, threshold 0.3, discard over 25
# preprocess17: exchange, threshold 0.3, discard over 30
# preprocess18: no exchange, threshold 0.3, discard over 30

# trun 430 total 480 freeze bert encoder layer1
# preprocess 15: exchange, threshold 0.3, no cut

# trun 447 total 480 epoch 2
# preprocess 19: no exchange, threshold 0.4, discard over 30

# #%%

#%%
QA_pred

# %%

