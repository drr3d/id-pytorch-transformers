"""
    This example intended as instruction for training new BERT model from scratch,
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
import pickle

from text_utils import TextDataset, loadAndCacheExamples
from model_utils import restoreModel
from tokenizer.tokenization_id import TokenizerId
from tokenizer.bert_prepare_inputdata import FullTokenizer as bertFullTokenizer, create_training_instances as bert_create_training_instances, InputFeatures as bertInputFeatures
from modeling.bert_modeling import BertModel, BertConfig


from torch.utils.data import DataLoader, RandomSampler, TensorDataset
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
               local_rank, train_batch_size, num_epoch, max_grad_norm, logging_steps,
               n_gpu=1, start_iters=0, mlm=False, save_dir='./pretrained/',  
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
    

    for cur_epoch in range(start_iters, num_epoch):
        start = time.time()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}".format(cur_epoch), disable=local_rank not in [-1, 0])

        loss = 0.
        for step, batch in enumerate(epoch_iterator):
                # The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
                #   To train the model, you should first set it back in training mode with ``model.train()``
                input_ids, input_mask, segment_ids = batch

                model.train()

                # https://github.com/huggingface/transformers/issues/1054
                outputs = model(input_ids.to(device), 
                                attention_mask=input_mask.to(device),
                                token_type_ids=segment_ids.to(device))

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
        op = "Epoch: {}, completed in: {}\n".format(cur_epoch, (end - start))
        print(op)
        with open("saved_trainingprogress.txt", 'a') as fw:
            fw.write(op)
        logging_loss = tr_loss

        # Save checkpoint
        _path = os.path.join(save_dir, 'epoch_{}-{}_id.ckpt'.format(cur_epoch, train_model_name))
        torch.save(model.state_dict(), _path)

def bertDataLoader(corpus_dir, trained_tensor_name, tokenizer_dir, 
                   spm_model_name, spm_vocab_name, 
                   do_lower_case=True):
    """ """
    print("Prepare BERT tokenizer...")
    tokenizer = bertFullTokenizer(piece_model=tokenizer_dir+spm_model_name,
                                  piece_vocab=tokenizer_dir+spm_vocab_name,
                                  do_lower_case=do_lower_case)
    
    print("Loading previously trained BERT tensor...")
    with open(corpus_dir+"/{}.pkl".format(trained_tensor_name), 'rb') as handle:
        train_data = pickle.load(handle)
    
    return train_data, tokenizer

def bertDataProcessing(corpus_dir, corpus_name, tokenizer_dir, spm_model_name, spm_vocab_name, 
                       do_lower_case=True, save_filename='bert_traintensor_wikiall', save_directory="./"):
    """ """
    print("Prepare BERT data prerpocessing...")
    max_seq_length = 128
    dupe_factor = 2 # Number of times to duplicate the input data (with different masks).
    short_seq_prob = 0.1 # Probability of creating sequences which are shorter than the maximum length.
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20 # Maximum number of masked LM predictions per sequence
    random_seed = 1337
    do_lower_case=do_lower_case # Whether to lower case the input text. Should be True for uncased models and False for cased models.

    print("Prepare BERT tokenizer...")
    tokenizer = bertFullTokenizer(piece_model=tokenizer_dir+spm_model_name,
                                  piece_vocab=tokenizer_dir+spm_vocab_name,
                                  do_lower_case=do_lower_case)

    print("Prepare BERT training instance...")
    instances = bert_create_training_instances([corpus_dir+corpus_name], tokenizer, max_seq_length, dupe_factor,
                                                short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                                                random.Random(random_seed))

    features = []
    print("Preparing training instance DONE...")
    print("Preparing append to TensorDataset...")
    for (inst_index, instance) in enumerate(tqdm(instances, desc="TensorDataset append")):
        input_ids = [ids for token in instance.tokens for ids in tokenizer.convert_tokens_to_ids(token)]
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)

        assert len(input_ids) <= max_seq_length
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = [ids for token in instance.masked_lm_labels for ids in tokenizer.convert_tokens_to_ids(token)]
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features.append(
                    bertInputFeatures(input_ids=input_ids,
                                        input_mask=input_mask,
                                        segment_ids=segment_ids,
                                        masked_lm_positions=masked_lm_positions,
                                        masked_lm_ids=masked_lm_ids,
                                        masked_lm_weights=masked_lm_weights,
                                        next_sentence_labels=[next_sentence_label]))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_masked_lm_positions = torch.tensor([f.masked_lm_positions for f in features], dtype=torch.long)
    all_masked_lm_ids = torch.tensor([f.masked_lm_ids for f in features], dtype=torch.long)
    all_masked_lm_weights = torch.tensor([f.masked_lm_weights for f in features], dtype=torch.float)
    all_next_sentence_labels = torch.tensor([f.next_sentence_labels for f in features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                                all_masked_lm_positions, all_masked_lm_ids,
                                all_masked_lm_weights, all_next_sentence_labels)
    

    print("saving data to: {}".format(save_directory))
    with open(save_directory+"/{}.pkl".format(save_filename), 'wb') as handle:
        pickle.dump(train_data, handle)

    return train_data, tokenizer

def main(corpus_dir, corpus_name, model_dir, trained_model_savedir, create_tokenizer=False, train_model_name='gpt2',
         train_spm=True, save_tokenized=False, dotraining=False, model_name=None, resume=False, vocab_name='vocab',
         resume_iters=0, spm_vocab_size=2000, spm_max_sentence_length=4098, spm_model_name='spm_id', block_size=512,
         spm_model_type='unigram', train_batch_size=1, num_epoch=1000, fp16=False, 
         trained_tensor_name='bert_traintensor_wikiall', tensor_from_pretrained=False):
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
    gradient_accumulation_steps = 5
    warmup_steps = 30

    tr_loss, logging_loss = 0.0, 0.0

    logging_steps = 50
    max_steps = 1000

    mlm_probability = 0.15
    local_rank = -1
    train_batch_size = train_batch_size
    block_size = block_size

    if not tensor_from_pretrained:
        ## create cache of training dataset
        train_dataset, tokenizer = bertDataProcessing(corpus_dir, corpus_name, model_dir, 
                                                    "{}.model".format(spm_model_name), "{}.vocab".format(spm_model_name), 
                                                    do_lower_case=True, save_filename='bert_traintensor_wikiall', 
                                                    save_directory="./")
    else:
        train_dataset, tokenizer = bertDataLoader(corpus_dir, tokenizer_dir=model_dir,
                                                    trained_tensor_name=trained_tensor_name,
                                                    spm_model_name="{}.model".format(spm_model_name),
                                                    spm_vocab_name="{}.vocab".format(spm_model_name)) 
    
    if dotraining:
        dataset = train_dataset
        print("Loading train_dataset done...")

        if max_steps > 0:
            t_total = max_steps
        else:
            t_total = len(dataset) // gradient_accumulation_steps * num_epoch
        print("t_total: {}".format(t_total))

        config = BertConfig(vocab_size_or_config_json_file=spm_vocab_size)
        model = BertModel(config)

        ## resume iters:
        if resume:
            model = restoreModel(model, resume_iters=resume_iters, 
                                model_name=model_name, 
                                model_save_dir=model_dir+trained_model_savedir, 
                                base_model_prefix='bert')
            
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
    """
    ## Training new data
    ## Step-1
    ##  set save_tokenized=True and create_tokenizer=True if you not yet do the training for tokenizers
    main(corpus_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/id/', 
         corpus_name='wiki_00mod_berts.txt', train_model_name='bert_id_wikicombinedAll',
         model_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/',
         trained_model_savedir="bert/", spm_model_name='spm_combinedAll_wordBert_id', 
         trained_tensor_name='bert_traintensor_wikiall', vocab_name=None, fp16=False,
         spm_vocab_size=100000,  spm_model_type='word', tensor_from_pretrained=True,
         save_tokenized=False, create_tokenizer=False, dotraining=True,  resume=False, 
         spm_max_sentence_length=80000, train_batch_size=6, num_epoch=100)

    """ 
    ## Training new data
    ## Step-1
    ##  set save_tokenized=True and create_tokenizer=True if you not yet do the training for tokenizers
    main(corpus_dir='../../Works/Data/ID/wiki_datasets/', 
         corpus_name='wiki_combinedall_ID_bert.txt', train_model_name='bert_id_wikicombinedAll_basehead_100k',
         model_dir='../../Works/Data/ID/wiki_datasets/',
         trained_model_savedir="bert/", spm_model_name='spm_combinedAll_wordBert_id', 
         trained_tensor_name='bert_traintensor_wikiall', vocab_name=None, fp16=True,
         spm_vocab_size=100000,  spm_model_type='word', tensor_from_pretrained=True,
         save_tokenized=False, create_tokenizer=False, dotraining=True,  resume=False, 
         spm_max_sentence_length=80000, train_batch_size=6, num_epoch=100)