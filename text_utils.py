import torch
import os
import pickle
import mmap

from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm

def getNumLines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def loadAndCacheExamples(train_data_file, block_size, tokenizer, evaluate=False, use_spm=False):
    dataset = TextDataset(tokenizer, file_path=train_data_file, block_size=block_size, use_spm=use_spm)
    return dataset

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
                print("Tokenize using spm")
                with open(file_path, encoding="utf-8") as f:
                    for line in tqdm(f, total=getNumLines(file_path)):
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
                    with open("tokenize_trainingprogress.txt", 'a') as fw:
                        fw.write("loop progress num {} - remaining data: {} - {}\n".format(i, len(tokenized_text), block_size))
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