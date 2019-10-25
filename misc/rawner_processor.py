import re
import sys

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
"""
# for raw_ner.txt
with open('iob_ner.txt', 'w', encoding='utf-8') as fo:
    with open('raw_ner.txt', 'r', encoding='utf-8') as fp:
       line = fp.readline()
       cnt = 1
       while line:
           m_entity = re.findall(" +TYPE=\"(.*?)\"", line.strip())
           m_entity_val = re.findall(r'<ENAMEX[^>]*>(.*?)</ENAMEX>', line.strip())
           m_clean = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', line.strip())
           
           _sentence_tag=[]
           _pair_tag=[]
           for ix, dt in enumerate(m_entity_val):
               approp_nertag=[]
               split_sent=[]
    
               for nix, n in enumerate(dt.split(' ')):
                   split_sent.append(n)
                   if m_entity[ix]=='PERSON':
                       if nix==0:
                           approp_nertag.append('B-PER')
                       else:
                           approp_nertag.append('I-PER')
                   elif m_entity[ix]=='LOCATION':
                       if nix==0:
                           approp_nertag.append('B-LOC')
                       else:
                           approp_nertag.append('I-LOC')
                   elif m_entity[ix]=='ORGANIZATION':
                       if nix==0:
                           approp_nertag.append('B-ORG')
                       else:
                           approp_nertag.append('I-ORG')
          
               _sentence_tag+=split_sent
               _pair_tag+=approp_nertag
           
           try:
               processed_tokenized = tokenizer.tokenize(m_clean)[:-1]
               for n in processed_tokenized:
                   token = re.sub(r'[▁]','',n)
                   if token in _sentence_tag:
                       fo.write("{} {}\n".format(token, _pair_tag[_sentence_tag.index(token)]))
                   else:
                       fo.write("{} {}\n".format(token, 'O'))
               fo.write("\n")
           except IndexError:
               print(_sentence_tag)
               print(_pair_tag)
               print(processed_tokenized)
               sys.exit()
           
           line = fp.readline()
           cnt += 1
"""
with open('iob_ner_2.txt', 'w', encoding='utf-8') as fo:
    with open('raw_ner_2.txt', 'r', encoding='utf-8') as fp:
       line = fp.readline()
       cnt = 1
       while line:
           m_entity = re.findall(r'<([^\W>]+)(\W|>)+', line.strip())
           m_entity = [ent[0] for ent in m_entity]

           m_entity_val = re.findall(r'<[^>]*>(.*?)</[^>]*>', line.strip())
           m_clean = re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', line.strip())

           _sentence_tag=[]
           _pair_tag=[]
           for ix, dt in enumerate(m_entity_val):
               approp_nertag=[]
               split_sent=[]
    
               for nix, n in enumerate(dt.split(' ')):
                   split_sent.append(n)
                   if m_entity[ix]=='PERSON':
                       if nix==0:
                           approp_nertag.append('B-PER')
                       else:
                           approp_nertag.append('I-PER')
                   elif m_entity[ix]=='LOCATION':
                       if nix==0:
                           approp_nertag.append('B-LOC')
                       else:
                           approp_nertag.append('I-LOC')
                   elif m_entity[ix]=='ORGANIZATION':
                       if nix==0:
                           approp_nertag.append('B-ORG')
                       else:
                           approp_nertag.append('I-ORG')
                   elif m_entity[ix]=='QUANTITY':
                       if nix==0:
                           approp_nertag.append('B-QTY')
                       else:
                           approp_nertag.append('I-QTY')
                   elif m_entity[ix]=='TIME':
                       if nix==0:
                           approp_nertag.append('B-TIME')
                       else:
                           approp_nertag.append('I-TIME')
          
               _sentence_tag+=split_sent
               _pair_tag+=approp_nertag
           
           try:
               processed_tokenized = tokenizer.tokenize(m_clean)[:-1]
               for n in processed_tokenized:
                   token = re.sub(r'[▁]','',n)
                   if token in _sentence_tag:
                       fo.write("{} {}\n".format(token, _pair_tag[_sentence_tag.index(token)]))
                   else:
                       fo.write("{} {}\n".format(token, 'O'))
               fo.write("\n")
           except IndexError:
               print(_sentence_tag)
               print(_pair_tag)
               print(processed_tokenized)
               sys.exit()
                       
           line = fp.readline()
           cnt += 1
