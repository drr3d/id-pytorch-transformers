"""
    This example intended as instruction for training new gpt2 model from scratch,
    aimed mainly for Indonesian language.
    But i thinks naturaly this model can be use to train another language as well.
"""
import sys
sys.path.append("..")

import os
import time
import torch
import torch.nn as nn
import numpy as np
import random
import math

from text_utils import TextDataset, loadAndCacheExamples
from model_utils import restoreModel
from tokenizer.tokenization_id import TokenizerId
from modeling.gpt2_modeling import GPT2Model, GPT2Config

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import WarmupLinearSchedule, AdamW


################################################################################################################
################                                 TRAINING                                 ######################
################################################################################################################
def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def doTraining(model, config, dataset, tokenizer, optimizer, scheduler, tr_loss, 
               logging_loss, gradient_accumulation_steps, mlm_probability, device, 
               local_rank, train_batch_size, num_epoch, max_grad_norm, n_gpu=1,
               logging_steps, start_iters=0, mlm=False,  save_dir='./pretrained/',  
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
        
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size)
    
    lm_loss = nn.Linear(config.n_embd, config.vocab_size, bias=True).to(device)

    for cur_epoch in range(start_iters, num_epoch):
        start = time.time()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}".format(cur_epoch), disable=local_rank not in [-1, 0])

        loss = 0.
        for step, batch in enumerate(epoch_iterator):
                # The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
                #   To train the model, you should first set it back in training mode with ``model.train()``
                inputs, labels = (batch.type(torch.cuda.LongTensor), batch.type(torch.cuda.LongTensor))
                inputs = inputs.to(device)
                labels = labels.to(device)

                model.train()
                outputs = model(inputs)
                
                logits = lm_loss(outputs[0])

                loss_fct = CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(logits.view(-1, logits.size(-1)),
                                labels.view(-1))

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


def main(corpus_dir, corpus_name, model_dir, trained_model_savedir, create_tokenizer=False, train_model_name='gpt2',
         train_spm=True, save_tokenized=False, dotraining=False, model_name=None, resume=False, vocab_name='vocab',
         resume_iters=0, spm_vocab_size=2000, spm_max_sentence_length=4098, spm_model_name='spm_id', block_size=512,
         spm_model_type='unigram', train_batch_size=1, num_epoch=1000, fp16=False):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--corpus_dir", default=None, type=str, required=True,
                        help="The directory where the corpus located.")
    parser.add_argument("--corpus_name", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--trained_model_savedir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--create_tokenizer", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--train_model_name", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--train_spm", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--save_tokenized", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
                    
    parser.add_argument("--dotraining", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--model_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--resume", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--vocab_name", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    args = parser.parse_args()

    ###################################################################################
    # set torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else: 
        device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    

    set_seed(seed=1332, n_gpu=n_gpu)

    num_epoch = num_epoch
    max_grad_norm = 1.0
    gradient_accumulation_steps = 50
    warmup_steps = 500

    tr_loss, logging_loss = 0.0, 0.0

    logging_steps = 50
    max_steps = 1000

    mlm_probability = 0.15
    local_rank = -1
    train_batch_size = train_batch_size
    block_size = block_size

    ## loading tokenizer
    tokenizer = TokenizerId(spm_vocab_size=spm_vocab_size)

    ## prepare dataset
    _dataset = corpus_dir + corpus_name
    if create_tokenizer:
        data_list=['<unk>','<sep>', '<cls>','<mask>']
        with open(_dataset, encoding="utf-8") as fp:
            line = fp.readline()
            while line:
               line = fp.readline()
               data_list.append(line)
        tokenizer.createVocab(data_list, spm_text_file=_dataset, data_dir=model_dir, train_spm=train_spm, 
                              spm_max_sentence_length=spm_max_sentence_length, spm_model_name=spm_model_name,
                              spm_model_type=spm_model_type)
    else:
        tokenizer.from_pretrained(model_dir, use_spm=train_spm, spm_model_name=spm_model_name, spm_max_sentence_length=spm_max_sentence_length,
                                    std_vocab_name=vocab_name)
    print("tokenizer.vocab_size: {}".format(tokenizer.vocab_size))

    ## saving tokenized object for consistent use
    if save_tokenized:
        tokenizer.save_pretrained(model_dir, vocab_name=vocab_name)

    ## create cache of training dataset
    train_dataset = loadAndCacheExamples(_dataset, block_size, tokenizer, evaluate=False, use_spm=train_spm)

    if dotraining:
        dataset = train_dataset
        print("Loading train_dataset done...")

        if max_steps > 0:
            t_total = max_steps
        else:
            t_total = len(dataset) // gradient_accumulation_steps * num_epoch
        print("t_total: {}".format(t_total))

        config = GPT2Config(vocab_size_or_config_json_file=tokenizer.vocab_size)

        # prepare output_attentions and hidden_states
        config.output_hidden_states=True

        model = GPT2Model(config)

        ## resume iters:
        if resume:
            model = restoreModel(model, resume_iters=resume_iters, model_name=model_name, model_save_dir=model_dir+trained_model_savedir, base_model_prefix='gpt2')
            
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

if __name__ == '__main__':
    
    ## Training new data
    ## Step-1
    ##  set save_tokenized=True and create_tokenizer=True if you not yet do the training for tokenizers
    main(corpus_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/id/', 
         corpus_name='combined_AE.txt', train_model_name='gpt2_id_wikicombinedAE',
         model_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/',
         spm_vocab_size=20000, vocab_name='vocab_wikicombindeAE_id', fp16=False,
         trained_model_savedir="gpt2/", spm_max_sentence_length=75000, spm_model_name='spm_wikicombindeAE_id',
         dotraining=True,  resume=False, train_spm=True, save_tokenized=False, create_tokenizer=False, block_size=768,
         spm_model_type='unigram', train_batch_size=1, num_epoch=10000)

    """
    main(corpus_dir='../../Data/ID/wiki_datasets/', model_name='epoch_2-gpt2_id_wikicombinedAE_id',
         corpus_name='combined_AE.txt', train_model_name='gpt2_id_wikicombinedAE',
         model_dir='../../Data/ID/wiki_datasets/', resume_iters=3,
         spm_vocab_size=20000, vocab_name='vocab_wikicombindeAE_id', fp16=True,
         trained_model_savedir="trained-model/gpt2/", spm_max_sentence_length=75000, spm_model_name='spm_wikicombindeAE_id',
         dotraining=True,  resume=True, train_spm=True, save_tokenized=False, create_tokenizer=False, block_size=768,
         spm_model_type='unigram', train_batch_size=3, num_epoch=10000)
    """