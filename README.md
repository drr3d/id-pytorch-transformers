<p align="center">
    <a href="https://github.com/drr3d/id-pytorch-transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/license-Apache%202-blue">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Natural Language Processing dengan <b>pytorch-transformers</b> bahasa Indonesia.
</h3>

# id-pytorch-transformers
Repository ini berisi modifikasi **[pytorch-transformers](https://github.com/huggingface/transformers)** (mungkin bukan modifikasi kali, tapi mayoritas copy-paste aja sih)
untuk beberapa tipe transformers network, dengan harapan mempermudah untuk mempelajari konstruksi tiap-tiap network nya dalam proses melakukan training dari awal(terutama untuk bahasa Indonesia).

sejauh yang saya temui pada pytorch-transformers, mungkin bisa saja melakukan proeses training dari awal, tapi saya tidak tau caranya(sepertinya mayoritas untuk proses **fine-tune**), 
jadi saya ambil bagian-bagian yang saya rasa penting untuk melakukan proses training dari awal. Sedangkan repo asli Author dari setiap tipe transformers, mayoritas menggunakan Tensorflow, sedangkan
apa yang saya bisa hanya PyTorch dan keras, karenanya saya pilih untuk menggunakan PyTorch.

> sebetulnya saya tidak terlalu yakin akan network re-kontruksi yang saya lakukan ini benar, tapi setelah dari beberapa percobaan, sepertinya bisa berjalan dengan baik.

untuk beberapa istilah yang saya gunakan, agar memperjelas:
1. **task** : task yang ada pada transformers diantaranya
    * GLUE
    * Text Generation
    * Squad
    * multiple-choice
    * etc..

    tapi dalam hal repository ini, task yang saya buat hanya:
    * Text Generation
    * NER
    * Text Classification
    * Question Answering
2. **Pre-trained**: proses training networks terhadap raw data/corpus dari awal.
3. **Fine-tuning**: selesai melakukan proses pre-trained pada raw data, kemudian dilakukan proses training ulang dengan tipe networks yang berbeda dengan data yang berbeda(bisa juga sama)
    > semua proses Pre-trained dikakukan dengan menggunakan BaseModel untuk setiap networks, misal untuk gpt2 yaitu **GPT2Model**, setelah itu selesai, misal anda ingin melakukan task
      **text-generation**, maka dilakukan proses training ulang dengan data baru(atau lama) dengan menggunakan tipe networks **GPT2LMHeadModel**.

    ```
      # for pre-trained
      model = GPT2Model()

      # ---------
      # for fine-tunine
      model = GPT2LMHeadModel()
    ```
      sebetulnya bisa saja anda lakukan langsung training dengan **GPT2LMHeadModel**, tapi nantinya hasil model tersebut sepertinya tidak bisa digunakan untuk jenis task yang lain.

## Data yang digunakan
* wiki-dump id (kurang lebih sekitar 480rb dokument)
* [Tempo online](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/tempo.zip)
* [Kompas online](http://ilps.science.uva.nl/ilps/wp-content/uploads/sites/6/files/bahasaindonesia/kompas.zip)
* [corpus frog story telling](https://github.com/davidmoeljadi/corpus-frog-storytelling)

hasil kombinasinya silahkan download [disini](https://drive.google.com/open?id=19h8W3OZwpML-OIBCp2lodBtwKRe0n6YI).

## Requirements

PyTorch `>=1.2.0`  
[pytorch-transformers](https://github.com/huggingface/transformers) `==1.2.0`  
[sentencepiece](https://github.com/google/sentencepiece)  


## Pre-trained model
proses training dilakukan di [Google-Colab](https://colab.research.google.com) serta laptop pribadi dengan GPU seadanya ya gaes. Training masih berlangsung sampai sekarang, 
karena itu hasil pretrained akan terus saya update.

```
### GPT2 config use: ###
n_positions=1024,
n_ctx=1024,
n_embd=600,
n_layer=12,
n_head=10,
resid_pdrop=0.1,
embd_pdrop=0.1,
attn_pdrop=0.1,
layer_norm_epsilon=1e-9,
initializer_range=0.02,

num_labels=1,
summary_type='cls_index',
summary_use_proj=True,
summary_activation=None,
summary_proj_to_labels=True,
summary_first_dropout=0.1

### XLNet config use: ###
d_model=600,
n_layer=12,
n_head=10,
d_inner=2048,
max_position_embeddings=300,
ff_activation="gelu",
untie_r=True,
attn_type="bi",

initializer_range=0.02,
layer_norm_eps=1e-12,

dropout=0.1,
mem_len=None,
reuse_len=None,
bi_data=False,
clamp_len=-1,
same_length=False,

finetuning_task=None,
num_labels=2,
summary_type='last',
summary_use_proj=True,
summary_activation='tanh',
summary_last_dropout=0.1,
start_n_top=5,
end_n_top=5

### BERT config use: ###
vocab_size_or_config_json_file=30522,
hidden_size=600,
num_hidden_layers=12,
num_attention_heads=10,
intermediate_size=2048,
hidden_act="gelu",
hidden_dropout_prob=0.1,
attention_probs_dropout_prob=0.1,
max_position_embeddings=512,
type_vocab_size=2,
initializer_range=0.02,
layer_norm_eps=1e-12
```

> Epoch saat ini: 2

hasil silahkan download disini:
* [XLNet(80jt parameter, 50k vocab, subword-unigram)]()
* [GPT2(80jt parameter, 50k vocab, subword-unigram)]()
* [BERT(107jt parameter, 100k vocab, subword-word)]()

## Networks tersedia
Networks yang tersedia diantaranya:
1. **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU)
2. **[GPT-2](https://blog.openai.com/better-language-models/)** (from OpenAI)
3. **[BERT](https://github.com/google-research/bert)** (from Google)

## Penjelasan Directory
1. [modeling](https://github.com/drr3d/id-pytorch-transformers/tree/master/modeling)
   file utama hasil modifikasi pemodelan networks.
2. [pretraining](https://github.com/drr3d/id-pytorch-transformers/tree/master/examples)
   file utama untuk melalukan proses pre-training.
3. [tokenizer](https://github.com/drr3d/id-pytorch-transformers/tree/master/tokenizer)
   file utama hasil modifikasi tokenizer yang digunakan setiap model. untuk Tokenizer, saya merubah semuanya dengan menggunakan SentencePiece
4. [tasks](https://github.com/drr3d/id-pytorch-transformers/tree/master/tasks)
   

### Memulai training:
modifikasi file dalam directory [pretraining](https://github.com/drr3d/id-pytorch-transformers/tree/master/examples), pilih file sesuai dengan tipe networks yang ingin
digunakan.

#### Masalah yang ditemui:
* https://github.com/NVIDIA/apex/issues/318
* https://discuss.pytorch.org/t/arguments-are-located-on-different-gpus/15372
* https://github.com/NVIDIA/apex/issues/341

***
# License
saya **tidak berhak mengakui** ini milik saya, karena pada dasarnya yang telah melakukan kerja dengan sangat baik adalah tim-tim dari HuggingFace dan para author asli yang memang pada awalnya membuat
dan berkontribusi sangat besar pada setiap jenis transformers network yang ada.

karena itu licensi mengikuti masing-masing pemilik kode aslinya. dalam hal ini karena mayoritas kode berasal dari **[pytorch-transformers](https://github.com/huggingface/transformers)**, maka lisensi
mengikuti mereka.
