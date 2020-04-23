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

DEVICE = 'cuda:0'
MODEL_SAVEPATH = './model_save/'

MAX_LENGTH = 512
DROPOUT_RATE = 0.1
BATCH_SIZE = 10
LEARNING_RATE = 5e-05
WEIGHT_DECAY = 1e-4
EPOCH = 10
THRESHOLD = 0.5

#%%
# read json file
import json
with open(TRAIN_ROUTE, 'r') as reader:
    train = json.loads(reader.read())
with open(VALID_ROUTE, 'r') as reader:
    valid = json.loads(reader.read())
with open(TEST_ROUTE, 'r') as reader:
    test = json.loads(reader.read())

train = train['data']
valid = valid['data']
test = test['data']

# %%
# get all paragraphs
train_paragraphs = []
valid_paragraphs = []
test_paragraphs = []

for data in train:
    train_paragraphs.append(data['paragraphs'])
for data in valid:
    valid_paragraphs.append(data['paragraphs'])
for data in test:
    test_paragraphs.append(data['paragraphs'])

#%%
# data preprocessing

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

    ids = 0
    non_match_count = 0
    matched_count = 0
    for paragraph in data_paragraphs:
        for sector in paragraph:
            context = sector['context']
            context = tokenizer.encode(context, add_special_tokens = False)
            for qa in sector['qas']:
                # prepare tokenized question + context
                question = tokenizer.encode(qa['question'], add_special_tokens = False)

                question_tokenized = tokenizer.batch_encode_plus([(question, context)],
                                                                max_length = MAX_LENGTH,
                                                                pad_to_max_length = True,
                                                                truncation_strategy='longest_first')

                # prepare answerable, start, stop
                if qa['answerable'] == False:
                    answerable = 0
                    answer_start = -1
                    answer_end = -1
                else:
                    answerable = 1
                    answer = tokenizer.encode(qa['answers'][0]['text'], add_special_tokens = False)

                    first_SEP_pos = np.where(np.array(question_tokenized['input_ids'][0])==102)[0]
                    match = find_sub_list(answer, question_tokenized['input_ids'][0][first_SEP_pos:])
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
# TODO
#print('preprocessing training dataset')
#test_question, test_ans = preprocessing_dataset(test_paragraphs, 'test', tokenizer)

# keys of qa set
# input_ids, encoded input
# token_type_ids, segment
# attention_mask, atta mask
# id, id
# answer, [answerable, answer_start, answer_end]

# %%
# define dataset
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

        if self.flag == 'train':
            for data in datas:
                batch_qc.append(data['input_ids'])
                batch_segments.append(data['token_type_ids'])
                batch_masks.append(data['attention_mask'])
                batch_answers.append(data['answer'])

            return torch.LongTensor(batch_qc), torch.LongTensor(batch_segments), torch.LongTensor(batch_masks), torch.LongTensor(batch_answers)
        else:
            for data in datas:
                batch_qc.append(data['input_ids'])
                batch_segments.append(data['token_type_ids'])
                batch_masks.append(data['attention_mask'])
    
            return torch.LongTensor(batch_qc), torch.LongTensor(batch_segments), torch.LongTensor(batch_masks)

#%%
# making torch Dataset
train_QA = QA_dataset(train_qa, 'train')
valid_QA = QA_dataset(valid_qa, 'train')
# TODO
# test_QA = QA_dataset(test_qa)

# take a look at ratio betwween answerable and non-answerable
yes_ans, no_ans = 0, 0
for i in train_QA:
    if i['answer'][0] == 1:
        yes_ans += 1
    else:
        no_ans += 1

print('Answerable counts vs Non-answerable counts (training data)')
print('Positive: ' + str(yes_ans))
print('Negative: ' + str(no_ans))

# %%
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class QA_Model(nn.Module):
    def __init__(self):
        super(QA_Model, self).__init__()

        self.num_labels = 513
        self.hidden_dim = 768
        self.vocab_size = 21128 # should not be used, just in case

        self.bert = BertModel.from_pretrained('bert-base-chinese')

        # for answerable start and end predict
        self.answerable_layer = nn.Sequential(
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
        )

        self.start_layer = nn.Sequential(
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1)
        )

        self.end_layer = nn.Sequential(
                nn.Dropout(DROPOUT_RATE),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,1)
        )

    def forward(self, qc, segment, mask):

        # sent BERT
        hidden_layer, output = self.bert(qc, segment, mask)
        # sent answerable start end
        answerable = self.answerable_layer(hidden_layer[:,0])
        ans_start = self.start_layer(hidden_layer[:,1:])
        ans_end = self.end_layer(hidden_layer[:,1:])

        return answerable, ans_start, ans_end


def get_matched_count(pred, gt):
    gt = gt.to(DEVICE)

    pred = torch.sigmoid(pred)
    pred = pred > THRESHOLD
    matched = (pred==gt).sum().data.item()

    return matched, len(pred)

def masked_out_noncontext(ans, segments):
    segments = segments[:,1:].float()
    ans = ans*segments
    print(ans)
    ans[ans == 0] = -float('inf')
    print(ans)

    return ans

#%%
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader

def _run_epoch(epoch, training):
    model.train(training)

    if training:
        desc = 'Train'
        dataset = train_QA
        shuffle = True
    else:
        desc = 'Valid'
        dataset = valid_QA
        shuffle = False

    dataloader = DataLoader(dataset = dataset,
                            batch_size = BATCH_SIZE,
                            shuffle = shuffle,
                            collate_fn = dataset.collate_fn,
                            num_workers = 4)
    
    trange = tqdm(enumerate(dataloader), total = len(dataloader), desc = desc)
    BCE_answerable_loss = 0
    CE_start_loss = 0
    CE_end_loss = 0
    answerable_match = 0
    cum_len = 0

    for step, (qcs, segments, masks, answers) in trange:

        answerable, ans_start, ans_end, loss_answerable, loss_start, loss_end = _run_iter(qcs, segments, masks, answers)

        if training:
            optimizer.zero_grad()
            batch_loss = loss_answerable + loss_start + loss_end
            batch_loss.backward()
            optimizer.step()

        # get loss
        BCE_answerable_loss += loss_answerable.item()
        CE_start_loss += loss_start.item()
        CE_end_loss += loss_end.item()
        # calculate Acc on answerable
        matched, batch_len = get_matched_count(answerable, answers[:,0])
        answerable_match += matched
        cum_len += batch_len

        trange.set_postfix(
            Answerable_acc = answerable_match / cum_len,
            Answerable_loss=BCE_answerable_loss / (step+1),
            Start_loss = CE_start_loss / (step+1),
            End_loss = CE_end_loss / (step+1)
        )

    if training:
        history['train'].append({'answerable_loss':BCE_answerable_loss/len(trange),
                                'start_loss':CE_start_loss/len(trange),
                                'end_loss':CE_end_loss/len(trange)}
                )
    else:
        history['valid'].append({'answerable_loss':BCE_answerable_loss/len(trange),
                        'start_loss':CE_start_loss/len(trange),
                        'end_loss':CE_end_loss/len(trange)}
                )


def _run_iter(qcs, segments, masks, answers):
    qcs = qcs.to(DEVICE)
    segments = segments.to(DEVICE) # int64
    masks = masks.to(DEVICE)
    answers = answers.to(DEVICE)

    batch_size = len(answers)
    answerable, ans_start, ans_end = model(qcs, segments, masks)
    answerable = answerable.squeeze(-1)
    ans_start = ans_start.squeeze(-1) # float 32
    ans_end = ans_end.squeeze(-1) # float 32

    gt_answerable = answers[:,0]
    gt_start = answers[:,1]
    gt_end = answers[:,2]
    
    gt_answerable = gt_answerable.float()
    # mask out ans_start, ans_end
    ans_start = masked_out_noncontext(ans_start, segments)
    ans_end = masked_out_noncontext(ans_end, segments)

    loss_answerable = answerable_criteria(answerable, gt_answerable)
    loss_start = start_criteria(ans_start, gt_start)
    loss_end = end_criteria(ans_end, gt_end)

    return answerable, ans_start, ans_end, loss_answerable, loss_start, loss_end


def save(epoch):
    if not os.path.exists(MODEL_SAVEPATH):
        os.makedirs(MODEL_SAVEPATH)
    
    torch.save(model.stat_dict(), MODEL_SAVEPATH+'model.epoch.'+str(epoch))
    with open(MODEL_SAVEPATH + 'history.json', 'w') as f:
        json.dump(history, f, indent = 4)


#%%
# setting up model
from transformers import AdamW

model = QA_Model()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr = LEARNING_RATE,
                  correct_bias = True   
            )

answerable_criteria = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.45))
start_criteria = torch.nn.CrossEntropyLoss(ignore_index = -1)
end_criteria = torch.nn.CrossEntropyLoss(ignore_index= -1)

model.to(DEVICE)
max_epoch = EPOCH
history = {'train':[],'valid':[]}

#%%
# training
for epoch in range(max_epoch):
    print('Epoch: {}'.format(epoch))
    _run_epoch(epoch, training = True)
    _run_epoch(epoch, training = False)
    save(epoch)

#%%
del model
torch.cuda.empty_cache()

# %%
a = [1,2,3,4,5,6,7,6,5,4,3,2,1,7]


# %%
