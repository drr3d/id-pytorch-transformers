import os
import sys
import math
import torch
import mmap
import random
import pickle
import time
import numpy as np

from modeling.gpt2_modeling import GPT2LMHeadModel, GPT2Config
from modeling.xlnet_modeling import XLNetLMHeadModel, XLNetConfig

from tokenization_id import TokenizerId

from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers import WarmupLinearSchedule, AdamW

from tqdm import tqdm, trange

def modelfrom_pretrained(model_use, model_config_use, model_dir, saved_modelname, vocab_size, device):
    ## Using for text generation
    config = model_config_use(vocab_size_or_config_json_file=vocab_size) # <-- make sure to use same vocab, if not would error and need to adjust vocab_size manually
    model = model_use(config)

    print("loading previous trained model...")
    state_dict = torch.load(model_dir + saved_modelname, map_location='cpu')

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ''
    model_to_load = model
    load(model_to_load, prefix=start_prefix)

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                        model.__class__.__name__, "\n\t".join(error_msgs)))

    # make sure word embedding weights are still tied
    model.tie_weights()
    model.eval()

    model.to(device)

    # Set model in evaluation mode to desactivate DropOut modules by default
    model.eval()
    print(model)
    return model

# https://datascience.stackexchange.com/questions/38540/are-there-any-good-out-of-the-box-language-models-for-python
# https://github.com/huggingface/transformers/issues/473
def perplexScore(sentence, tokenizers, models, device, use_spm=False):
    if not use_spm:
        tokenize_input = tokenizers.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    else:
        tensor_input = torch.tensor([tokenizers.convert_tokens_to_ids(sentence, is_spm=use_spm)])
    with torch.no_grad():
        loss=models(tensor_input.to(device), labels=tensor_input.to(device))

    return math.exp(loss[0])

perp_test_sent=['Raja Bhumibol Adulyadej dari Thailand',
                'kemenangan untuk mahasiswa atas berhasilnya penolakan RUU kuhp kpk',
                'there is a book in the desk']


################################################################################################################
################                                 TRAINING                                 ######################
################################################################################################################
def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512, use_spm=False, add_special_tokens=False):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')

        if os.path.exists(cached_features_file):
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            print("Creating features from dataset file at %s", directory)

            self.examples = []
            tokenized_text = None
            print("convert token to id...")
            if use_spm:
                tokenized_text = []
                with open(file_path, encoding="utf-8") as f:
                    for line in tqdm(f, total=get_num_lines(file_path)):
                        tokenized_text+=tokenizer.convert_tokens_to_ids(line, is_spm=use_spm)
            else:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text, is_spm=use_spm))

            print("len(tokenized_text): {} - block size: {}".format(len(tokenized_text), block_size))
            print("Approximatly will run on {} loops.".format(int(len(tokenized_text)/block_size)))
            tmp_run_on=[]
            for i in range(0, len(tokenized_text), block_size):
                tmp_run_on.append(i)
            i=0
            print("begin appending data...")
            while len(tokenized_text) >= block_size:  # Truncate in block of block_size
                if add_special_tokens:
                    self.examples.append(tokenizer.add_special_tokens_single_sequence(tokenized_text[:block_size]))
                else:
                    self.examples.append(tokenized_text[:block_size])
                tokenized_text = tokenized_text[block_size:]
                i+=1
                if i in tmp_run_on:
                    print("loop progress num {} - remaining data: {} - {}".format(i, len(tokenized_text), block_size))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

def load_and_cache_examples(train_data_file, block_size, tokenizer, evaluate=False, use_spm=False):
    dataset = TextDataset(tokenizer, file_path=train_data_file, block_size=block_size, use_spm=use_spm)
    return dataset

def restore_model(model, resume_iters, model_name, model_save_dir):
    """Restore the trained generator and discriminator."""

    model_path = os.path.join(model_save_dir, '{}.ckpt'.format(model_name))
    print('Loading the trained models from step {} - of file: {}'.format(resume_iters, model_path))

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    #model.zero_grad() # <-- no need, unless fine-tune
    return model

def doTraining(model, dataset, tokenizer, optimizer, scheduler, tr_loss, 
               logging_loss, gradient_accumulation_steps, mlm_probability, device, 
               local_rank, train_batch_size, num_epoch, max_grad_norm,
               logging_steps, start_iters=0, mlm=False,  save_dir='./pretrained/',  
               train_model_name='gpt2'):

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
                
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Update parameters and take a step using the computed gradient
                    optimizer.step()

                    # Update learning rate schedule
                    scheduler.step()

                    # Clear out the gradients (by default they accumulate)
                    model.zero_grad()
    
        end = time.time()
        op = "Epoch: {}, completed in: {}, loss: {}, perplexity: {}\n".format(cur_epoch, (end - start), (tr_loss - logging_loss)/logging_steps, 
                                                                              perplexScore(perp_test_sent[1], tokenizer, model, 'cuda', use_spm=True))
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
         spm_model_type='unigram'):
    ###################################################################################
    # set torch device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else: 
        device = torch.device("cpu")
    n_gpu = torch.cuda.device_count()

    set_seed(seed=1332, n_gpu=n_gpu)

    num_epoch = 100
    max_grad_norm = 1.0
    gradient_accumulation_steps = 50
    warmup_steps = 0

    tr_loss, logging_loss = 0.0, 0.0

    logging_steps = 50
    max_steps = -1

    mlm_probability = 0.15
    local_rank = -1
    train_batch_size = 1
    block_size = block_size

    ## loading tokenizer
    tokenizer = TokenizerId(spm_vocab_size=spm_vocab_size)

    ## prepare dataset
    _dataset = corpus_dir + corpus_name
    if create_tokenizer:
        data_list=['<unk>','<sep>', '<cls>']
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
    train_dataset = load_and_cache_examples(_dataset, block_size, tokenizer, evaluate=False, use_spm=train_spm)

    if dotraining:
        dataset = train_dataset
        print("Loading train_dataset done...")

        if max_steps > 0:
            t_total = max_steps
        else:
            t_total = len(dataset) // gradient_accumulation_steps * num_epoch
        print("t_total: {}".format(t_total))

        ## Prepare model and training
        models = [
                    (GPT2LMHeadModel, GPT2Config), 
                    (XLNetLMHeadModel, XLNetConfig)
                ]
        config = models[0][1](vocab_size_or_config_json_file=tokenizer.vocab_size)
        model = models[0][0](config)

        ## resume iters:
        if resume:
            model = restore_model(model, resume_iters=resume_iters, model_name=model_name, model_save_dir=model_dir+trained_model_savedir)
            
        model.to(device)

        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of model_parameters: {}".format(num_params))

        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

        doTraining(model, train_dataset, tokenizer, optimizer, scheduler, tr_loss, logging_loss, 
                   gradient_accumulation_steps, mlm_probability, device, local_rank, train_batch_size,
                   num_epoch=num_epoch, start_iters=resume_iters, max_grad_norm=max_grad_norm, 
                   logging_steps=logging_steps, save_dir=model_dir+trained_model_savedir, train_model_name=train_model_name)

if __name__ == '__main__':
    
    
    ## Training new data
    ## Step-1
    
    main(corpus_dir='./samples/wiki_datasets/id/', corpus_name='wiki_00_mod.txt', train_model_name='gpt2_id_wiki00mod',
         model_dir='./samples/wiki_datasets/trained_model/', spm_vocab_size=2000, vocab_name='vocab_wiki00mod_id',
         trained_model_savedir="gpt2/", spm_max_sentence_length=25000, spm_model_name='spm_wiki00mod_id',
         dotraining=True,  resume=False, train_spm=True, save_tokenized=True, create_tokenizer=True, block_size=256,
         spm_model_type='unigram')
    """
    ## Step-2 (optional, only if there is an error, and you unwilling to train the vocab again)
    main(corpus_dir='./samples/wiki_datasets/id/', corpus_name='wiki_00_mod.txt', train_model_name='gpt2_id_wiki00mod',
         model_dir='./samples/wiki_datasets/trained_model/', spm_vocab_size=2000, vocab_name='vocab_wiki00mod_id',
         trained_model_savedir="gpt2/", spm_max_sentence_length=25000, spm_model_name='spm_wiki00mod_id',
         dotraining=True,  resume=False, train_spm=True, save_tokenized=False, create_tokenizer=False, block_size=256,
         spm_model_type='unigram')
    """

    """
    ## Resume training
    main(corpus_dir='./samples/wiki_datasets/id/', corpus_name='combined_AE.txt', train_model_name='gpt2_id_combinedAE',
         model_dir='./samples/wiki_datasets/trained_model/', spm_vocab_size=50000, vocab_name='vocab_combinedAE_id',
         trained_model_savedir="gpt2/", spm_max_sentence_length=80000, spm_model_name='spm_combinedAE_unigram_id',
         dotraining=True,  resume=True, train_spm=True, save_tokenized=False, create_tokenizer=False, block_size=512,
         spm_model_type='unigram', model_name='epoch_1-gpt2_id_combinedAE_id')
    """

    """ 
    ## Only process tokenizer
    ##  set save_tokenized=True, create_tokenizer=True for retraining the tokenizer
    main(corpus_dir='./samples/wiki_datasets/id/', corpus_name='combined_all.txt', train_model_name='gpt2_id_combinedAll',
         model_dir='./samples/wiki_datasets/trained_model/', spm_vocab_size=150000, vocab_name='vocab_combinedAll_id',
         trained_model_savedir="gpt2/", spm_max_sentence_length=80000, spm_model_name='spm_combinedAll_unigram_id',
         dotraining=False,  resume=False, train_spm=True, save_tokenized=True, create_tokenizer=True, block_size=512)
    """