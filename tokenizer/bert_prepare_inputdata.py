"""
    based on original comment in file https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L185
    , thats for BERT, we need to prepare our WIKI-id dataset into specific format where:
    1. one sentence per-line
    2. blank line after document

    WIKI-id data has arbitrary len of text per-line, so we need to do some cleaning to support above format.
"""
from tqdm import tqdm
import mmap
import random
import six
import sentencepiece as spm
import collections
import torch

from torch.utils.data import TensorDataset

#corpus_dir='../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/id/'
#corpus_name = 'combined_all.txt' # combined_all.txt' #
#spm_retrained_corpus = 'combined_all.txt'

#tokenizer_dir = '../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/'
#spm_model_name = 'spm_combinedAll_wordBert_id.model'
#spm_vocab_name = 'spm_combinedAll_wordBert_id.vocab'

def getNumLines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def recleanWiki(corpus_dir, corpus_name, output_filename):
    raw_data = []
    with open(corpus_dir+corpus_name, encoding="utf-8") as f:
        for line in tqdm(f, total=getNumLines(corpus_dir+corpus_name)):
            # i dont want to make it hard how to get sentence properly, just split it by .
            #  then process sentence only if have len > 5
            sent_data = [sent_on_aline.strip() for sent_on_aline in line.split('.') if len(sent_on_aline.strip().split(' ')) > 5]

            # append only docs that have more than 1 sentence
            if len(sent_data) > 1:
                raw_data.append(sent_data)

    # write re-formated data
    with open(output_filename, 'w',  encoding="utf-8") as file_handler:
        print("Write output file...")
        for n in tqdm(raw_data, total=len(raw_data)):
            for sent in n:
                file_handler.write("{}\n".format(sent.strip()))
            file_handler.write("\n")
#recleanWiki('wiki_00mod_bert.txt')


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in tqdm(input_files, desc="Create TrainingInstance"):
        with open(input_file, "rb") as reader:
            while True:
                line = convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, 
                                 vocab_words, rng,
                                 do_whole_word_mask=False):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                        max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                    # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "rb") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class SentencePieceTokenizer(object):
    """Runs Google's SentencePiece tokenization."""
    def __init__(self, model, unk_token="<unk>"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model)
        self.unk_token = unk_token

    def tokenize(self, text):
        output_ids = self.sp.EncodeAsIds(text)
        output_tokens = [convert_to_unicode(self.sp.IdToPiece(i))
                        if i != 0 else self.unk_token
                        for i in output_ids]
        return output_tokens

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "rb") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            token = token.split('\t')[0]
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class FullTokenizer(object):
    """Runs end-to-end tokenziation."""
    # using sentenpiece only

    def __init__(self, piece_model, piece_vocab, do_lower_case=True):
        self.vocab = load_vocab(piece_vocab)
        self.sentencepiece_tokenizer = SentencePieceTokenizer(model=piece_model)

    def tokenize(self, text):
        split_tokens = []
        text = ' '.join(whitespace_tokenize(text))
        split_tokens = self.sentencepiece_tokenizer.tokenize(text)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return self.sentencepiece_tokenizer.sp.EncodeAsIds(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.sentencepiece_tokenizer.sp.IdToPiece(ids)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids=None, 
                 masked_lm_weights=None, next_sentence_labels=None,
                 label_id=None, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights
        self.next_sentence_labels = next_sentence_labels

        # for NER
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

# spm_model_type: 'unigram', 'bpe'
create_spm_model = False
if create_spm_model:
    import sentencepiece as spm
    spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={} \
                                    --hard_vocab_limit=false --max_sentence_length={} \
                                    --user_defined_symbols=[MASK], --control_symbols=[CLS],[SEP]'.format(corpus_dir+spm_retrained_corpus, 'spm_combinedAll_wordBert_id', 100000, 'word', 80000))
"""
max_seq_length = 128
dupe_factor = 10
short_seq_prob = 0.1
masked_lm_prob = 0.15
max_predictions_per_seq = 20
random_seed = 1337
do_lower_case = True


tokenizer = FullTokenizer(piece_model=tokenizer_dir+spm_model_name,
                          piece_vocab=tokenizer_dir+spm_vocab_name,
                          do_lower_case=do_lower_case)


instances = create_training_instances(
    [corpus_dir+corpus_name], tokenizer, max_seq_length, dupe_factor,
    short_seq_prob, masked_lm_prob, max_predictions_per_seq,
    random.Random(random_seed))




features = []
for (inst_index, instance) in enumerate(instances):
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
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              masked_lm_positions=masked_lm_positions,
                              masked_lm_ids=masked_lm_ids,
                              masked_lm_weights=masked_lm_weights,
                              next_sentence_labels=[next_sentence_label]))


import sys

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
all_masked_lm_positions = torch.tensor([f.masked_lm_positions for f in features], dtype=torch.long)
all_masked_lm_ids = torch.tensor([f.masked_lm_ids for f in features], dtype=torch.long)
all_masked_lm_weights = torch.tensor([f.masked_lm_weights for f in features], dtype=torch.float)
all_next_sentence_labels = torch.tensor([f.next_sentence_labels for f in features], dtype=torch.long)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, 
                            all_masked_lm_positions, all_masked_lm_ids,
                            all_next_sentence_labels)


from torch.utils.data import DataLoader, RandomSampler
batchsize=1
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize)

epoch_iterator = tqdm(train_dataloader, desc="Iteration-{}".format(1))
nn=0
for step, batch in enumerate(epoch_iterator):
    print(batch)

    if nn == 3:
        sys.exit()
    nn+=1
 """ 