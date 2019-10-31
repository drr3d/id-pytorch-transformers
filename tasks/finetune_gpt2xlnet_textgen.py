#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import argparse
import math
import time
import os
import logging
from tqdm import trange
import torch.nn as nn

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler
import numpy as np

from transformers import WarmupLinearSchedule, AdamW
from text_utils import TextDataset, loadAndCacheExamples

from tqdm import tqdm, trange

try:
    from id_pytorch_transformers.model_utils import restoreModel
    from id_pytorch_transformers.tokenizer.tokenization_id import TokenizerId
    from id_pytorch_transformers.modeling.xlnet_modeling import XLNetConfig, XLNetLMHeadModel
    from id_pytorch_transformers.modeling.gpt2_modeling import GPT2LMHeadModel, GPT2Config
except ImportError:
    sys.path.append("..")
    from model_utils import restoreModel
    from tokenizer.tokenization_id import TokenizerId
    from modeling.xlnet_modeling import XLNetConfig, XLNetLMHeadModel
    from modeling.gpt2_modeling import GPT2LMHeadModel, GPT2Config, GPT2Model


# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ Nama Menteri Enggar muncul seusai KPK meringkus Bowo Sidik dan menetapkannya sebagai tersangka suap dan gratifikasi. Kepada penyidik, 
Bowo mengaku menerima uang dari berbagai sumber, salah satunya dari Menteri Perdagangan.

Enggar diduga memberi Bowo Sidik uang sebesar Rp 2 miliar agar ia mengamankan Peraturan Menteri Perdagangan Nomor 16
tentang Perdagangan Gula Kristal Rafinasi Melalui Pasar Lelang Komoditas, yang akan berlaku akhir Juni 2017."""


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.9, is_xlnet=False, device='cpu', target_model='xlnet'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        #for _ in trange(length):
        for _ in range(length):
            inputs = {'input_ids': generated}

            if target_model == 'xlnet':
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def txtGen(input, model_name_or_path=None, length=50, padding_text="", 
         temperature=1.0, top_k=0, top_p=0.9, seed=42, target_model='xlnet',
         use_spm=True, spm_vocab_size=2000, spm_model_name='spm_id', n_embd=128,
         vocab_model_dir='./samples/wiki_datasets/trained_model/', pretrained_model_dir=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    set_seed(seed, n_gpu)

    tokenizer = TokenizerId(spm_vocab_size=spm_vocab_size)
    tokenizer.from_pretrained(vocab_model_dir, use_spm=use_spm, spm_model_name=spm_model_name)
    print("tokenizer process is finish...")

    # Models with memory likes to have a long prompt for short inputs.
    if target_model=='xlnet':
        raw_text = (padding_text if padding_text else PADDING_TEXT) + input
    else:
        raw_text = input

    def procContextTokens(tokens):
        return tokenizer.encode(raw_text, is_spm=use_spm)
    context_tokens = procContextTokens(raw_text)
    print("context_tokens: {} - len: {}".format(context_tokens, len(context_tokens)))
    print(tokenizer.decode(context_tokens, clean_up_tokenization_spaces=True, use_spm=use_spm))

    if target_model=='xlnet':
        # Instantiate model.
        # XLNet
        config = XLNetConfig(vocab_size_or_config_json_file=tokenizer.vocab_size) # <-- make sure to use same vocab, if not would error and need to adjust vocab_size manually
        model = XLNetLMHeadModel(config)
    elif target_model=='gpt2':
        # GPT-2
        config = GPT2Config(vocab_size_or_config_json_file=tokenizer.vocab_size, n_embd=n_embd) # <-- make sure to use same vocab, if not would error and need to adjust vocab_size manually
        model = GPT2LMHeadModel(config) # GPT2Model(config) #

    print("loading previous trained model...")
    model = restoreModel(model, resume_iters=None, 
                        model_name=model_name_or_path, 
                        model_save_dir=pretrained_model_dir, 
                        from_pretrained=True)

    model.to(device)

    # Set model in evaluation mode to desactivate DropOut modules by default
    model.eval()
    print(model)
    
    raw_ct = context_tokens
    print("\ngpt2-text generation processing input: \n> {} ...\n {}".format(input, "-"*100))
    for i in range(0, 10):
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            target_model=target_model
        )
        out = out[0, len(context_tokens):].tolist()
        #context_tokens+=out

        text = tokenizer.decode(out, clean_up_tokenization_spaces=True, use_spm=use_spm)
        print("{}: {}".format(i, text))

finetune = True
if finetune:
    def doTraining(model, config, dataset, tokenizer, optimizer, scheduler, tr_loss, 
               logging_loss, gradient_accumulation_steps, mlm_probability, device, 
               local_rank, train_batch_size, num_epoch, max_grad_norm, logging_steps,
               n_gpu=1, start_iters=0, mlm=False,  save_dir='./pretrained/',  
               train_model_name='gpt2', fp16=False):

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            #  Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'], defaul 01
            print("Trained using apex fp16..")
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        print("n_gpu: {}".format(n_gpu))
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)

        for cur_epoch in range(start_iters, num_epoch):
            start = time.time()
            epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}".format(cur_epoch), disable=local_rank not in [-1, 0])

            for step, batch in enumerate(epoch_iterator):
                    # The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
                    #   To train the model, you should first set it back in training mode with ``model.train()``
                    inputs, labels = (batch.type(torch.cuda.LongTensor), batch.type(torch.cuda.LongTensor))
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    model.train()
                    outputs = model(inputs, labels=labels)

                    loss = outputs[0]
 
                    if n_gpu > 1:
                        loss = loss.mean()

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
            op = "Epoch: {}, completed in: {}, loss: {}, perplexity: {}\n".format(cur_epoch, (end - start), (tr_loss - logging_loss)/logging_steps, 
                                                                                math.exp(loss))
            print(op)
            with open("saved_trainingprogress.txt", 'a') as fw:
                fw.write(op)
            logging_loss = tr_loss

            # Save checkpoint
            _path = os.path.join(save_dir, 'epoch_{}-{}_id.ckpt'.format(cur_epoch, train_model_name))
            torch.save(model.state_dict(), _path)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else: 
        device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    

    set_seed(seed=1332, n_gpu=n_gpu)

    max_grad_norm = 1.0
    gradient_accumulation_steps = 5
    warmup_steps = 30

    tr_loss, logging_loss = 0.0, 0.0

    logging_steps = 5
    max_steps = 1000

    mlm_probability = 0.15
    local_rank = -1
    train_batch_size = 1
    block_size = 600

    corpus_dir = '../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/id/'
    model_dir = '../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/'
    trained_model_savedir = 'gpt2/'
    corpus_name = 'wiki_00_mod.txt'
    model_name = 'epoch_7-gpt2_id_wikimod00_2kvocab_lmhead_id_id' #'epoch_11-gpt2_id_wikicombinedAll_basehead_50k_id'

    vocab_name = 'vocab_combinedAll_id'
    spm_model_name = 'spm_combinedAll_unigram_id'
    spm_vocab_size = 50000
    spm_max_sentence_length=50000

    train_model_name = 'gpt2_id_wikimod00_2kvocab_lmhead_id'
    resume = True
    train_spm = True
    resume_iters = 8
    num_epoch = 100
    fp16 = False

    ## loading tokenizer
    tokenizer = TokenizerId(spm_vocab_size=spm_vocab_size)

    ## prepare dataset
    _dataset = corpus_dir + corpus_name
    
    tokenizer.from_pretrained(model_dir, use_spm=train_spm, spm_model_name=spm_model_name, spm_max_sentence_length=spm_max_sentence_length,
                                std_vocab_name=vocab_name)
    print("tokenizer.vocab_size: {}".format(tokenizer.vocab_size))


    ## create cache of training dataset
    train_dataset = loadAndCacheExamples(_dataset, block_size, tokenizer, evaluate=False, use_spm=train_spm)

    dataset = train_dataset
    print("Loading train_dataset done...")

    if max_steps > 0:
        t_total = max_steps
    else:
        t_total = len(dataset) // gradient_accumulation_steps * num_epoch
    print("t_total: {}".format(t_total))

    config = GPT2Config(vocab_size_or_config_json_file=tokenizer.vocab_size, n_embd=600)

    # prepare output_attentions and hidden_states
    config.output_hidden_states=True
    model = GPT2LMHeadModel(config)

    ## resume iters:
    if resume:
        ## set is_finetune and from_pretrained False if you resume train from previously fine-tuned model,
        ##     set True otherwise
        model = restoreModel(model, resume_iters=resume_iters, model_name=model_name, 
                            model_save_dir=model_dir+trained_model_savedir, 
                            is_finetune=False, from_pretrained=False,
                            base_model_prefix='gpt2', )
        
    model.to(device)

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of model_parameters: {}".format(num_params))

    weight_decay = 0.1
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=0.00025, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    doTraining(model, config, train_dataset, tokenizer, optimizer, scheduler, tr_loss, logging_loss, 
                gradient_accumulation_steps, mlm_probability, device, local_rank, train_batch_size, n_gpu=n_gpu,
                num_epoch=num_epoch, start_iters=resume_iters, max_grad_norm=max_grad_norm, fp16=fp16,
                logging_steps=logging_steps, save_dir=model_dir+trained_model_savedir, train_model_name=train_model_name)
    
else:
    text = ['Pesta Olahraga Asia Tenggara', "Tolondadu merupakan salah satu", "ketika lapar maka kita akan memesan"]
    txtGen(text[0], model_name_or_path='epoch_7-gpt2_id_wikimod00_2kvocab_lmhead_id_id', spm_vocab_size=50000, length=25, use_spm=True,
            spm_model_name='spm_combinedAll_unigram_id',  target_model='gpt2',  temperature=1.0, top_k=0, top_p=0.9, n_embd=600,
            vocab_model_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/',
            pretrained_model_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/gpt2/')