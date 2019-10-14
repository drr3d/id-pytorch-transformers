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

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from tokenization_id import TokenizerId
from modeling.xlnet_modeling import XLNetConfig, XLNetLMHeadModel
from modeling.gpt2_modeling import GPT2LMHeadModel, GPT2Config

from tqdm import tqdm, trange

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
        for _ in trange(length):
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


def main(input, model_name_or_path=None, length=50, padding_text="", 
         temperature=1.0, top_k=0, top_p=0.9, seed=42, target_model='xlnet',
         use_spm=True, spm_vocab_size=2000, spm_model_name='spm_id',
         vocab_model_dir='./samples/wiki_datasets/trained_model/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    set_seed(seed, n_gpu)

    #_dataset = './samples/wiki_datasets/id/wiki_00_mod.txt'
    """with open(_dataset, encoding="utf-8") as fp:
        line = fp.readline()
        while line:
            line = fp.readline()
            data_list.append(line)
    print("datalist append is finish...")"""
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
        config = GPT2Config(vocab_size_or_config_json_file=tokenizer.vocab_size, n_embd=384) # <-- make sure to use same vocab, if not would error and need to adjust vocab_size manually
        model = GPT2LMHeadModel(config)

    print("loading previous trained model...")
    state_dict = torch.load(model_name_or_path, map_location='cpu')

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

    for i in range(0, 3):
        #print("context_tokens :{}".format(context_tokens))
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            target_model='gpt2'
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True, use_spm=use_spm)
        print(text)
    
    
main(input='saya mau sukses dengan', model_name_or_path='./samples/wiki_datasets/trained_model/gpt2/epoch_35-gpt2_id_combinedAE_id.ckpt',
     temperature=5.0, top_k=40, top_p=0.9, target_model='gpt2', spm_vocab_size=50000, length=50, spm_model_name='spm_combinedAE_unigram_id')