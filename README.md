<p align="center">
    <a href="https://github.com/drr3d/id-pytorch-transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/license-Apache%202-blue">
    </a>
</p>

# id-pytorch-transformers
Pretrained pemodelan bahasa Indonesia dengan **[pytorch-transformers](https://github.com/huggingface/transformers)**.

Repository ini berisi modifikasi **[pytorch-transformers](https://github.com/huggingface/transformers)** untuk beberapa tipe transformers network,
dengan harapan mempermudah untuk mempelajari konstruksi tiap-tiap network nya dalam proses training dari awal(terutama untuk bahasa Indonesia).

sejauh yang saya temui pada pytorch-transformers, mungkin bisa saja melakukan proeses training dari awal, tapi saya tidak tau caranya(sepertinya mayoritas untuk proses **fine-tune**), 
jadi saya ambil bagian-bagian yang saya rasa penting untuk melakukan proses training dari awal.

saya tidak yakin sih sebetulnya network re-kontruksi yang saya lakukan ini benar, tapi setelah dari beberapa percobaan, sepertinya bisa berjalan dengan baik.

untuk beberapa istilah yang saya gunakan, agar memperjelas:
1. **task** : task yang ada pada transformers diantaranya
    * GLUE
    * Text Generation
    * Squad
    * multiple-choice
    * etc..
2. **Pre-trained**: proses training networks akan raw data/corpus dari awal.
3. **Fine-tuning**: selesai melakukan proses pre-trained pada raw data, kemudian dilakukan proses training ulang dengan tipe networks yang berbeda dengan data yang berbeda(bisa juga sama)
    > semua proses Pre-trained dikakukan dengan menggunakan BaseModel untuk setiap networks, misal untuk gpt2 yaitu **GPT2Model**, setelah itu selesai, misal anda ingin melakukan task
      **text-generation**, maka dilakukan proses training ulang dengan data baru(atau lama) dengan menggunakan tipe networks **GPT2LMHeadModel**.

      sebetulnya bisa saja anda lakukan langsung training dengan **GPT2LMHeadModel**, tapi nantinya hasil tersebut sepertinya tidak bisa digunakan untuk jenis task yang lain.

# Networks tersedia
Networks yang tersedia diantaranya:
1. **[XLNet](https://github.com/zihangdai/xlnet/)** (from Google/CMU)
2. **[GPT-2](https://blog.openai.com/better-language-models/)** (from OpenAI)
3. **[BERT](https://github.com/google-research/bert)** (from Google) -- **IN PROGRESS**

# Penjelasan Directory
1. modeling
2. examples

***
# License
saya **tidak berhak mengakui** ini milik saya, karena pada dasarnya yang telah melakukan kerja dengan sangat baik adalah tim-tim dari HuggingFace dan lainnya yang memang pada awalnya membuat jenis-jenis
transformers network yang ada.

karena itu licensi mengikuti masing-masing pemilik kode aslinya. dalam hal ini karena mayoritas kode berasal dari **[pytorch-transformers](https://github.com/huggingface/transformers)**, maka lisensi
mengikuti mereka.
