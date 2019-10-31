# -*- coding: utf-8 -*-
import sys
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm, trange
from transformers import WarmupLinearSchedule, AdamW
from seqeval.metrics import classification_report

import numpy as np
import random
import logging
import torch.nn.functional as F

try:
    from id_pytorch_transformers.bert_modeling import BertForTokenClassification, BertConfig
except ImportError:
    sys.path.append("..")
    from model_utils import restoreModel

    from modeling.bert_modeling import BertForTokenClassification, BertConfig
    from tokenizer.bert_prepare_inputdata import convert_examples_to_features, NerProcessor, FullTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

max_seq_length = 256
train_batch_size = 1
num_train_epochs = 100
gradient_accumulation_steps = 1
max_grad_norm = 1.0
fp16 = False
spm_vocab_size=50000
warmup_steps = 30
logging_steps = 50
max_steps = 1000
t_total = max_steps



def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def doEval(data_dir, max_seq_length, label_list, tokenizer, model, eval_batch_size=1):
    eval_examples = ner_processor.get_test_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)}

    xx=0
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
        #print("logits: {}".format(logits))
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        if xx<=2:
            print(input_ids)
            print(label_ids)
            print("----")
            print(logits)
            xx+=1

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j,m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
    print(y_true[:2])
    print(y_pred[:2])
    report = classification_report(y_true, y_pred,digits=3)
    logger.info("\n%s", report)

if torch.cuda.is_available():
    device = torch.device('cuda')
else: 
    device = torch.device("cpu")
n_gpu = torch.cuda.device_count()

set_seed(seed=1332, n_gpu=n_gpu)


finetune = False
if finetune:
    model_dir = '../../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/'
    spm_model_name = 'spm_combinedAll_lcase_uni50k_id.model'
    spm_vocab_name = 'spm_combinedAll_lcase_uni50k_id.vocab'
    trained_model_savedir = 'bert/'
    model_name = 'epoch_15-bert_id_wikicombinedAll_tokenclasshead_lcase_uni50k_id_id'#'epoch_8-bert_id_wikicombinedAll_basehead_lcase_uni50k_id'
    do_lower_case=True
    eval_only = True
    is_finetune = False # set True for the first time finetume is executed, for new set False for resume training

    ner_processor = NerProcessor()
    label_list = ner_processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = FullTokenizer(piece_model=model_dir+spm_model_name,
                                piece_vocab=model_dir+spm_vocab_name,
                                do_lower_case=do_lower_case)

    train_examples = ner_processor.get_train_examples('../misc/')
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)


    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    class RunningAverage():
        """A simple class that maintains the running average of a quantity

        Example:
        ```
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg() = 3
        ```
        """

        def __init__(self):
            self.steps = 0
            self.total = 0

        def update(self, val):
            self.total += val
            self.steps += 1

        def __call__(self):
            L = 0.
            try:
                L = self.total / float(self.steps)
                return L
            except ZeroDivisionError as ze: 
                return 0.

    class Ner(BertForTokenClassification):
        def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
            batch_size,max_len,feat_dim = sequence_output.shape
            valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                        if valid_ids[i][j].item() == 1:
                            jj += 1
                            valid_output[i][jj] = sequence_output[i][j]
            sequence_output = self.dropout(valid_output)
            logits = self.classifier(sequence_output)

            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                attention_mask_label = None
                if attention_mask_label is not None:
                    active_loss = attention_mask_label.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
            else:
                return logits

    print("num_labels: {}".format(num_labels))
    config = BertConfig(vocab_size_or_config_json_file=spm_vocab_size, num_labels=num_labels, hidden_size=600, num_attention_heads=12, intermediate_size=2048)


    #model = Ner(config)
    model = BertForTokenClassification(config)
    model = restoreModel(model, resume_iters=0, 
                            model_name=model_name, 
                            model_save_dir=model_dir+trained_model_savedir, 
                            base_model_prefix='bert',
                            from_pretrained=True,
                            is_finetune=is_finetune)
                            
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


    loss_avg = RunningAverage()

    global_step = 0
    if eval_only:
        doEval('../misc/', max_seq_length, label_list, tokenizer, model, eval_batch_size=2)
    else:
        for eph in range(int(num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration-{}".format(eph))):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                
                model.train()
                
                #loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                loss = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=label_ids)
                #print(loss)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                #print(loss)
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                loss_avg.update(loss.item())
                
                #print(input_ids[0].tolist())
                #print([tokenizer.convert_ids_to_tokens(n) for n in input_ids[0].tolist()])
                #print(label_ids)
                print(loss_avg())
                #if step==3:
                #    sys.exit()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            if eph%0==0:
                train_model_name = 'bert_id_wikicombinedAll_tokenclasshead_lcase_uni50k_id'
                _path = os.path.join('/content/drive/My Drive/iPad/bert/', 'epoch_{}-{}_id.ckpt'.format(cur_epoch, train_model_name))
                torch.save(model.state_dict(), _path)

                doEval('../misc/', max_seq_length, label_list, tokenizer, model, eval_batch_size=6)
else:
    class BertNer(BertForTokenClassification):

        def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None):
            sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
            batch_size,max_len,feat_dim = sequence_output.shape
            valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda' if torch.cuda.is_available() else 'cpu')
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                        if valid_ids[i][j].item() == 1:
                            jj += 1
                            valid_output[i][jj] = sequence_output[i][j]
            sequence_output = self.dropout(valid_output)
            logits = self.classifier(sequence_output)
            return logits

    class Ner:

        def __init__(self, model_dir: str, model_name=None, outmedia=None):
            self.model , self.tokenizer, self.model_config = self.load_model(model_dir, model_name=model_name, outmedia=outmedia)
            self.label_map = self.model_config["label_map"]
            self.max_seq_length = self.model_config["max_seq_length"]
            self.label_map = {int(k):v for k,v in self.label_map.items()}
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()

        def load_model(self, model_dir: str, model_config: str = "model_config.json", model_name=None, outmedia=None):
            model_config = os.path.join(model_dir,model_config)
            model_config = json.load(open(model_config))
            model = BertNer.from_pretrained(model_dir)
            tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=model_config["do_lower"])
            
            if model_name is not None:
                if outmedia is not None:
                    with outmedia:
                        print("Loading predict model from: {}".format(model_dir+"/"+model_name))
                        state_dict = torch.load(model_dir+"/"+model_name, map_location=lambda storage, loc: storage)
                        model.load_state_dict(state_dict)
                        
            return model, tokenizer, model_config

        def tokenize(self, text: str):
            """ tokenize input"""
            words = word_tokenize(text)
            tokens = []
            valid_positions = []
            for i,word in enumerate(words):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                for i in range(len(token)):
                    if i == 0:
                        valid_positions.append(1)
                    else:
                        valid_positions.append(0)
            return tokens, valid_positions

        def preprocess(self, text: str):
            """ preprocess """
            tokens, valid_positions = self.tokenize(text)
            ## insert "[CLS]"
            tokens.insert(0,"[CLS]")
            valid_positions.insert(0,1)
            ## insert "[SEP]"
            tokens.append("[SEP]")
            valid_positions.append(1)
            segment_ids = []
            for i in range(len(tokens)):
                segment_ids.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            print(tokens)
            print(input_ids)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid_positions.append(0)
            return input_ids,input_mask,segment_ids,valid_positions

        def predict(self, text: str):
            input_ids,input_mask,segment_ids,valid_ids = self.preprocess(text)
            input_ids = torch.tensor([input_ids],dtype=torch.long,device=self.device)
            input_mask = torch.tensor([input_mask],dtype=torch.long,device=self.device)
            segment_ids = torch.tensor([segment_ids],dtype=torch.long,device=self.device)
            valid_ids = torch.tensor([valid_ids],dtype=torch.long,device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask,valid_ids)
            logits = F.softmax(logits,dim=2)
            logits_label = torch.argmax(logits,dim=2)
            logits_label = logits_label.detach().cpu().numpy().tolist()[0]

            logits_confidence = [values[label].item() for values,label in zip(logits[0],logits_label)]

            logits = []
            pos = 0
            for index,mask in enumerate(valid_ids[0]):
                if index == 0:
                    continue
                if mask == 1:
                    logits.append((logits_label[index-pos],logits_confidence[index-pos]))
                else:
                    pos += 1
            logits.pop()

            labels = [(self.label_map[label],confidence) for label,confidence in logits]
            words = word_tokenize(text)
            assert len(labels) == len(words)
            output = [{"word":word,"tag":label,"confidence":confidence} for word,(label,confidence) in zip(words,labels)]
            return output

    model = Ner(self.torchmodel_dir.path, model_name=standardized_fn, outmedia=out)
    output = model.predict(globals() ['predictText_{}'.format(standardized_fn)].value)