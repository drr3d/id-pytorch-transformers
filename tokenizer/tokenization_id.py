# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Tokenization classes for XLNet model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os
import re
import torch
from shutil import copyfile, move
import pickle
import warnings

from sklearn.feature_extraction.text import CountVectorizer
import sentencepiece as spm

import unicodedata
import six

logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = u'▁'

class TokenizerId(object):
    """
        SentencePiece based tokenizer. Peculiarities:

            - requires `SentencePiece <https://github.com/google/sentencepiece>`_
    """

    def __init__(self, unk_token="<unk>", sep_token="<sep>",
                 pad_token="<pad>", cls_token="<cls>", spm_vocab_size = 2000,
                 do_lower_case=True, remove_space=True, subword_method='sklearn'):
        super(TokenizerId, self).__init__()

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space

        self.vocabulary = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}
        
        self.subword_method = subword_method

        self._sep_token = sep_token
        self._pad_token = pad_token
        self._cls_token = cls_token
        self._unk_token = unk_token

        # Added tokens
        self.added_tokens_encoder = [sep_token, pad_token, cls_token, unk_token, ' ']

        self.special_tokens_map = {}
        self.spm_vocab_size = spm_vocab_size
        self.use_spm=False

    @property
    def vocab_size(self):
        if self.use_spm:
            return self.spm_vocab_size
        else:
            return len(self.vocabulary)

    def createVocab(self, doc, data_dir='', spm_text_file='', train_spm=True, spm_max_sentence_length=4098, spm_model_type='unigram', spm_model_name='spm_id'):
        if self.vocab_size <=0:
            if self.subword_method=='sklearn':
                # sklearn
                self.skl_model = CountVectorizer(token_pattern=r"""\d+|[-a-zA-Z]+|['".,:^%@#$&\*!()]+|<\s*[^>]*>""")
                self.skl_model.fit_transform(doc)

                self.vocabulary = self.skl_model.vocabulary_
                self.unk_vocab_ix = self.vocabulary.get(self._unk_token)
                self.sklearn_tokenizer = self.skl_model.build_tokenizer()

                if os.path.isdir(data_dir):
                    if train_spm:
                        # spm_model_type: 'unigram', 'bpe'
                        spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={} --hard_vocab_limit=false --max_sentence_length={}'.format(spm_text_file, spm_model_name, self.spm_vocab_size, 
                                                                                                                                                                                spm_model_type, spm_max_sentence_length))
                        print("Moving trained spm model: {} into directory: {}".format("{}.model".format(spm_model_name), data_dir))
                        move("{}.model".format(spm_model_name), data_dir)

                        print("Moving trained spm vocab: {} into directory: {}".format("{}.vocab".format(spm_model_name), data_dir))
                        move("{}.vocab".format(spm_model_name), data_dir)
                    self.use_spm=True
                    self.sp_model = spm.SentencePieceProcessor()
                    self.sp_model.Load(data_dir + "/{}.model".format(spm_model_name))
                    #print("change spm vocab size from {} into {}".format(self.spm_vocab_size, spm_max_sentence_length))
                    #self.spm_vocab_size = spm_max_sentence_length
                else:
                    print("data_dir is invalid!! spm would not be use...")
        else:
            print("Seems vocabulary has been loaded...")

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if six.PY2 and isinstance(outputs, str):
            outputs = outputs.decode('utf-8')

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    # tokenize the doc and lemmatize its tokens
    def idTokenizer(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_tokens(tok_list, text):
            tokenized_text = self.sklearn_tokenizer(text)
            return [token for token in tokenized_text if token not in tok_list]

        added_tokens = self.added_tokens_encoder + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text

    def _tokenize(self, text, return_unicode=True, sample=False):
        """ Tokenize a string.
            return_unicode is used only for py2
        """
        text = self.preprocess_text(text)
        # note(zhiliny): in some systems, sentencepiece only accepts str for py2
        if six.PY2 and isinstance(text, unicode):
            text = text.encode('utf-8')

        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        # note(zhiliny): convert back to unicode for py2
        if six.PY2 and return_unicode:
            ret_pieces = []
            for piece in new_pieces:
                if isinstance(piece, str):
                    piece = piece.decode('utf-8')
                ret_pieces.append(piece)
            new_pieces = ret_pieces

        return new_pieces

    def tokenize(self, text, return_unicode=True, sample=False, use_spm=False):
        """ Tokenize a string.
            return_unicode is used only for py2
        """
        text = self.preprocess_text(text)
        if use_spm:
            return self._tokenize(' '.join(self.idTokenizer(text)))
        else:
            return self.idTokenizer(text)

    def encode(self,
                text,
                text_pair=None,
                add_special_tokens=False,
                max_length=None,
                stride=0,
                truncate_first_sequence=True,
                return_tensors=None,
                is_spm=False,
                **kwargs):
        """
        Converts a string in a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defined the number of additional tokens.
            truncate_first_sequence: if there is a specified max_length, this flag will choose which sequence
                will be truncated.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """
        encoded_inputs = self.encode_plus(text,
                                          text_pair=text_pair,
                                          max_length=max_length,
                                          add_special_tokens=add_special_tokens,
                                          stride=stride,
                                          truncate_first_sequence=truncate_first_sequence,
                                          return_tensors=return_tensors,
                                          is_spm=is_spm,
                                          **kwargs)

        return encoded_inputs["input_ids"]

    def encode_plus(self,
                    text,
                    text_pair=None,
                    add_special_tokens=False,
                    max_length=None,
                    stride=0,
                    truncate_first_sequence=True,
                    return_tensors=None,
                    is_spm=False,
                    **kwargs):
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional informations:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defined the number of additional tokens.
            truncate_first_sequence: if there is a specified max_length, this flag will choose which sequence
                will be truncated.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """

        def get_input_ids(text):
            if isinstance(text, six.string_types):
                if is_spm:
                    return self.convert_tokens_to_ids(text, is_spm=is_spm)
                else:
                    return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], six.string_types):
                return self.convert_tokens_to_ids(text, is_spm=is_spm)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None
        
        return self.prepare_for_model(first_ids,
                                      pair_ids=second_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncate_first_sequence=truncate_first_sequence,
                                      return_tensors=return_tensors,
                                      is_spm=is_spm)


    def prepare_for_model(self, ids, pair_ids=None, max_length=None, add_special_tokens=False, stride=0,
                          truncate_first_sequence=True, return_tensors='pt', is_spm=False):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
        It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens and manages a window stride for
        overflowing tokens

        Args:
            ids: list of tokenized input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            pair_ids: Optional second list of input ids. Can be obtained from a string by chaining the
                `tokenize` and `convert_tokens_to_ids` methods.
            max_length: maximum length of the returned list. Will truncate by taking into account the special tokens.
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            stride: window stride for overflowing tokens. Can be useful for edge effect removal when using sequential
                list of inputs.
            truncate_first_sequence: if set to `True` and an optional second list of input ids is provided,
                alongside a specified `max_length`, will truncate the first sequence if the total size is superior
                than the specified `max_length`. If set to `False`, will truncate the second sequence instead.
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.

        Return:
            a dictionary containing the `input_ids` as well as the `overflowing_tokens` if a `max_length` was given.
        """
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        if max_length:
            n_added_tokens = 0
            if pair and n_added_tokens + (len_pair_ids if truncate_first_sequence else len_ids) >= max_length:
                logger.warning(
                    "You supplied a pair of sequence in which the sequence that will not be truncated is longer than the maximum specified length."
                    "This pair of sequences will not be truncated.")
            else:
                if n_added_tokens + len_ids + len_pair_ids > max_length:
                    if truncate_first_sequence or not pair:
                        encoded_inputs["overflowing_tokens"] = ids[max_length - len_pair_ids - n_added_tokens - stride:]
                        ids = ids[:max_length - len_pair_ids - n_added_tokens]
                    elif not truncate_first_sequence and pair:
                        encoded_inputs["overflowing_tokens"] = pair_ids[max_length - len_ids - n_added_tokens - stride:]
                        pair_ids = pair_ids[:max_length - len_ids - n_added_tokens]
                    else:
                        logger.warning(
                            "Cannot truncate second sequence as it is not provided. No truncation.")

        if add_special_tokens:
            sequence = self.add_special_tokens_sequence_pair(ids, pair_ids) if pair else self.add_special_tokens_single_sequence(ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids) if pair else [0] * len(sequence)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_tensors == 'pt':
            sequence = torch.tensor([sequence])
            token_type_ids = torch.tensor([token_type_ids])
        elif return_tensors is not None:
            logger.warning("Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(return_tensors))
        #print("sequence: {}".format(sequence))
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids

        return encoded_inputs
    
    @staticmethod
    def clean_up_tokenization(out_string):
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ','
                        ).replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m").replace(" do not", " don't"
                        ).replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re")
        return out_string

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True, use_spm=False):
        """
        Converts a sequence of ids (integer) in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.
        Similar to doing ``self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))``.
        """
        #print("token_ids: {}".format(token_ids))
        filtered_tokens = self._convert_id_to_token(token_ids, skip_special_tokens=skip_special_tokens, use_spm=use_spm)
        #print("filtered_tokens: {}".format(filtered_tokens))
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append("")
            else:
                current_sub_text.append(token)

        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ''.join(sub_texts)

        if self._sep_token is not None and self._sep_token in text:
            text = text.replace(self._cls_token, self._sep_token)
            split_text = list(filter(lambda sentence: len(sentence) > 0, text.split(self._sep_token)))
            if clean_up_tokenization_spaces:
                clean_text = [self.clean_up_tokenization(text) for text in split_text]
                return clean_text
            else:
                return split_text
        else:
            if clean_up_tokenization_spaces:
                clean_text = self.clean_up_tokenization(text)
                return clean_text
            else:
                return text

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = list(self.convert_tokens_to_ids(t) for t in all_toks)
        return all_ids

    def convert_tokens_to_ids(self, tokens, is_spm=False):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens is None:
            return None
        ids = None
        if not is_spm:
            ids = []
            for token in tokens:
                ids.append(self.vocabulary.get(token, self.unk_vocab_ix))
        else:
            # take whole sentence, no need to tokenized it first
            ids = self.sp_model.EncodeAsIds(tokens)
        return ids

    def _convert_id_to_token(self, ids, skip_special_tokens=False, use_spm=False):
        if isinstance(ids, int):
            return self.convert_id_to_token(ids, use_spm=use_spm)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self.convert_id_to_token(index, use_spm=use_spm))
        return tokens

    def convert_id_to_token(self, index, return_unicode=True, skip_special_tokens=False, use_spm=False):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        if use_spm: # spm in this case
            token = self.sp_model.IdToPiece(index)
        else:
            try:
                token = list(self.vocabulary.keys())[list(self.vocabulary.values()).index(index)]
            except ValueError as e:
                token = self._unk_token
                pass

        if six.PY2 and return_unicode and isinstance(token, str):
            token = token.decode('utf-8')
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = ' '.join(tokens).replace(SPIECE_UNDERLINE, '').strip()
        return out_string

    def add_special_tokens_single_sequence(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        An XLNet sequence has the following format: X [SEP][CLS]
        """
        sep = [self.vocabulary.get(self._sep_token)]
        cls = [self.vocabulary.get(self._cls_token)]
        return token_ids + sep + cls

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        An XLNet sequence pair has the following format: A [SEP] B [SEP][CLS]
        """

        sep = [self.vocabulary.get(self._sep_token)]
        cls = [self.vocabulary.get(self._cls_token)]
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 2
        | first sequence    | second sequence     | CLS segment ID
        """
        sep = [self.vocabulary.get(self._sep_token)]
        cls = [self.vocabulary.get(self._cls_token)]
        cls_segment_id = [2]

        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def from_pretrained(self, load_directory, spm_text_file='', train_spm=False, use_spm=False, spm_model_type='unigram',
                        spm_max_sentence_length=4098, spm_model_name='spm_id', std_vocab_name='vocab'):
        if not os.path.isdir(load_directory):
            logger.error("Loading directory ({}) should be a directory".format(load_directory))
            return
        
        if std_vocab_name is not None:
            print("loading: {}".format(load_directory+"/{}.pkl".format(std_vocab_name)))
            with open(load_directory+"/{}.pkl".format(std_vocab_name), 'rb') as handle:
                self.skl_model = pickle.load(handle)

            # summarize
            self.vocabulary = self.skl_model.vocabulary_
            self.unk_vocab_ix = self.vocabulary.get(self._unk_token)
            self.sklearn_tokenizer = self.skl_model.build_tokenizer()
        else:
            warnings.warn("Warning vocab name is empty! This only allowed on BERT training..")
    
        if use_spm:
            if train_spm:
                # model_type: 'unigram', 'bpe'
                spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={} --hard_vocab_limit=false --max_sentence_length={}'.format(spm_text_file, spm_model_name, 
                                                                                                                                                                        self.spm_vocab_size, spm_model_type, spm_max_sentence_length))
            self.sp_model = spm.SentencePieceProcessor()
            print("Loading spm on: {}".format(load_directory + "/{}.model".format(spm_model_name)))
            self.sp_model.Load(load_directory + "/{}.model".format(spm_model_name))
            self.use_spm=True
            #print("change spm vocab size from {} into {}".format(self.spm_vocab_size, spm_max_sentence_length))
            #self.spm_vocab_size = spm_max_sentence_length


    def save_pretrained(self, save_directory, vocab_name='vocab'):
        """ Save the tokenizer vocabulary files together with:
                - added tokens,
                - special-tokens-to-class-attributes-mapping,
                - tokenizer instantiation positional and keywords inputs (e.g. do_lower_case for Bert).

            This won't save modifications other than (added tokens and special token mapping) you may have
            applied to the tokenizer after the instantion (e.g. modifying tokenizer.do_lower_case after creation).

            This method make sure the full tokenizer can then be re-loaded using the :func:`~transformers.PreTrainedTokenizer.from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            logger.error("Saving directory ({}) should be a directory".format(save_directory))
            return
        print("saving data to: {}".format(save_directory))
        if vocab_name is not None:
            with open(save_directory+"/{}.pkl".format(vocab_name), 'wb') as handle:
                pickle.dump(self.skl_model, handle)
        else:
            warnings.warn("Warning vocab name is empty! This only allowed on BERT training..")


"""
# create vocab:##
_dataset = './tests_samples/wiki_datasets/id/combined_AE_split.txt'
data_list=[]#["<unk>", "<sep>", "<cls>"]
with open(_dataset, encoding="utf-8") as fp:
    line = fp.readline()
    while line:
       line = fp.readline()
       data_list.append(line)

tokenizer_id = XLNetTokenizerId(vocab_file=None)

#tokenizer_id.from_pretrained('./tests_samples/wiki_datasets/trained_model/')
tokenizer_id.createVocab(data_list, spm_text_file=_dataset, data_dir='./', train_spm=False, spm_max_sentence_length=60000, spm_vocab_size=150000)
print(tokenizer_id.vocab_size)
print("\n\n")
#print(tokenizer_id._tokenize(text='Asam deoksiribonukleat, lebih dikenal dengan<sep>singkatan DNA (bahasa Inggris: <unk>'))
#tokenized_text = tokenizer_id.tokenize('Asam deoksiribonukleat, lebih dikenal dengan<sep>singkatan DNA (bahasa Inggris: <unk>', use_spm=True)

text_to_test = ['Asam deoksiribonukleat, lebih dikenal dengan', 'where do the spirit goes woekrokwpe fijoir 神奇隧道']
print("given text: {}".format(text_to_test[1]))
print("given special token: {}".format("None"))#"<unk> , <sep> , <cls>"))
print("\n")
print("using sencentepiece:\n--------------------")
tokenized_text = tokenizer_id.tokenize(text_to_test[1], use_spm=True)
print(tokenized_text)
print("_convert_id_to_token: {}".format(tokenizer_id._convert_id_to_token(tokenizer_id.convert_tokens_to_ids(text_to_test[1], is_spm=True), use_spm=True)))
print(tokenizer_id.convert_tokens_to_ids(text_to_test[1], is_spm=True))
#print(tokenizer_id.convert_tokens_to_ids('Asam deoksiribonukleat, lebih dikenal dengan<sep>singkatan DNA (bahasa Inggris: <unk>', is_spm=True))

print("\nusing std tokenizer:\n--------------------")
tokenized_text = tokenizer_id.tokenize(text_to_test[1], use_spm=False)
print(tokenized_text)
print(tokenizer_id.convert_tokens_to_ids(tokenized_text, is_spm=False))

#tokenizer_id.save_pretrained('./tests_samples/wiki_datasets/trained_model/')
"""