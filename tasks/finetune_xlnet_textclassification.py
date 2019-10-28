# -*- coding: utf-8 -*-
import sys
import pandas as pd
import math
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
import random
import time

import torch
import os
from tqdm import tqdm,trange
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from transformers import WarmupLinearSchedule, AdamW

try:
    from id_pytorch_transformers.model_utils import restoreModel
    from id_pytorch_transformers.tokenizer.tokenization_id import TokenizerId
    from id_pytorch_transformers.modeling.xlnet_modeling import XLNetConfig, XLNetLMHeadModel
    from id_pytorch_transformers.modeling.gpt2_modeling import GPT2LMHeadModel, GPT2Config
except ImportError:
    sys.path.append("..")
    from model_utils import restoreModel
    from tokenizer.tokenization_id import TokenizerId
    from modeling.xlnet_modeling import XLNetConfig, XLNetForSequenceClassification

# positive data will be give 1 as label
df_pos = pd.read_fwf('../../temporary_before_move_to_git/id-pytorch-transformers/samples/twitter/positive_id.txt', header=None, names=['sentence'])
df_pos['label'] = 1


# positive data will be give 1 as label
df_neg = pd.read_fwf('../../temporary_before_move_to_git/id-pytorch-transformers/samples/twitter/negative_id.txt', header=None, names=['sentence'])
df_neg['label'] = 0

df_train = pd.concat([df_pos, df_neg])
print(len(df_train[df_train['label']==1]))

## prepare for input
# Get sentence data
sentences = df_train.sentence.to_list()
print(sentences[0])

# Get tag labels data
labels = df_train.label.to_list()
print(labels[0])

tag2idx={'0': 0, '1': 1}
# Mapping index to name
tag2name={tag2idx[key] : key for key in tag2idx.keys()}

# Make label into id
tags = [tag2idx[str(lab)] for lab in labels]
print(tags[0])


## prepare tokenizers
vocab_model_dir = '../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/'
spm_model_name = 'spm_wikicombindeAE_id'
spm_vocab_size=20000
tokenizer = TokenizerId(spm_vocab_size=spm_vocab_size)
tokenizer.from_pretrained(vocab_model_dir, use_spm=True, spm_model_name=spm_model_name)

## prepare transformers pre-trained models
config = XLNetConfig(vocab_size_or_config_json_file=tokenizer.vocab_size) # <-- make sure to use same vocab, if not would error and need to adjust vocab_size manually
model = XLNetForSequenceClassification(config)

## prepare text input
max_len  = 64

full_input_ids = []
full_input_masks = []
full_segment_ids = []

SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

UNK_ID = tokenizer.encode("<unk>", is_spm=True)[0] # tokenizer.convert_tokens_to_ids("<unk>", is_spm=True) #
CLS_ID = tokenizer.encode("<cls>", is_spm=True)[0]
SEP_ID = tokenizer.encode("<sep>", is_spm=True)[0]
MASK_ID = tokenizer.encode("<mask>", is_spm=True)[0]
EOD_ID = tokenizer.encode("<eod>", is_spm=True)[0]

for i,sentence in enumerate(sentences):
    # Tokenize sentence to token id list
    tokens_a = tokenizer.encode(sentence, is_spm=True)
    
    # Trim the len of text
    if(len(tokens_a)>max_len-2):
        tokens_a = tokens_a[:max_len-2]
        
        
    tokens = []
    segment_ids = []
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
        
    # Add <sep> token 
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)
    
    
    # Add <cls> token
    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    
    input_ids = tokens
    
    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length at fornt
    if len(input_ids) < max_len:
        delta_len = max_len - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len
    
    full_input_ids.append(input_ids)
    full_input_masks.append(input_mask)
    full_segment_ids.append(segment_ids)
    
    if 3 > i:
        print("No.:%d"%(i))
        print("sentence: %s"%(sentence))
        print("input_ids:%s"%(input_ids))
        print("attention_masks:%s"%(input_mask))
        print("segment_ids:%s"%(segment_ids))
        print("\n")


def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def doTraining(model, dataset, tokenizer, optimizer, scheduler, tr_loss, 
               logging_loss, gradient_accumulation_steps, mlm_probability, device, 
               local_rank, train_batch_size, num_epoch, max_grad_norm,
               logging_steps, start_iters=0, mlm=False,  save_dir='./pretrained/',  
               train_model_name='gpt2', fp16=True):

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #  Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'], defaul 01
        print("Trained using apex fp16..")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)

    for cur_epoch in range(start_iters, num_epoch):
        start = time.time()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}".format(cur_epoch), disable=local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
                # The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
                #   To train the model, you should first set it back in training mode with ``model.train()``
                
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_segs,b_labels = batch

                model.train()
                outputs = model(input_ids=b_input_ids, 
                                token_type_ids=b_segs,
                                input_mask=b_input_mask,
                                labels=b_labels)
                
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    if fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Update parameters and take a step using the computed gradient
                    optimizer.step()

                    # Update learning rate schedule
                    scheduler.step()

                    # Clear out the gradients (by default they accumulate)
                    model.zero_grad()
    
        end = time.time()
        op = "Epoch: {}, completed in: {}, loss: {}\n".format(cur_epoch, (end - start), (tr_loss - logging_loss)/logging_steps)
        print(op)
        with open("saved_trainingprogress.txt", 'a') as fw:
            fw.write(op)
        logging_loss = tr_loss

        # Save checkpoint
        _path = os.path.join(save_dir, 'epoch_{}-{}_id.ckpt'.format(cur_epoch, train_model_name))
        torch.save(model.state_dict(), _path)

## split data
tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(full_input_ids, tags,full_input_masks,full_segment_ids, random_state=4, test_size=0.3)
print(len(tr_inputs),len(val_inputs),len(tr_segs),len(val_segs))

## prepare torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)

# Set batch num
batch_num = 4

# Set token embedding, attention embedding, segment embedding
train_dataset = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
#train_sampler = RandomSampler(train_data)

# Drop last can make batch training better for the last one
#train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

model_name_or_path='epoch_1-xlnet_id_wikicombindeAE_id'
pretrained_model_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/xlnet/'
model = restoreModel(model, resume_iters=None, 
                    model_name=model_name_or_path, 
                    model_save_dir=pretrained_model_dir,
                    is_finetune=True,
                    from_pretrained=True)
# Set model to GPU,if you are using GPU machine
model.to(device)

num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of model_parameters: {}".format(num_params))

set_seed(seed=1332, n_gpu=n_gpu)

num_epoch = 10000
max_grad_norm = 1.0
gradient_accumulation_steps = 50
warmup_steps = 500

tr_loss, logging_loss = 0.0, 0.0

logging_steps = 50
max_steps = 20000

mlm_probability = 0.15
local_rank = -1
train_batch_size = 1
block_size = 512

if max_steps > 0:
    t_total = max_steps
else:
    t_total = len(dataset) // gradient_accumulation_steps * num_epoch
print("t_total: {}".format(t_total))
    
optimizer = AdamW(model.parameters(), lr=0.00025, weight_decay=0.01)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

doTraining(model, train_dataset, tokenizer, optimizer, scheduler, tr_loss, logging_loss, 
            gradient_accumulation_steps, mlm_probability, device, local_rank, train_batch_size,
            num_epoch=num_epoch, start_iters=0, max_grad_norm=max_grad_norm, fp16=False,
            logging_steps=logging_steps, save_dir=pretrained_model_dir, train_model_name='xlnet_id_twitterSentiment')

valid_data = TensorDataset(val_inputs, val_masks,val_segs, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)