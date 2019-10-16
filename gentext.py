from tasks.transformers_textgen import *
txtGen(input='Indonesia',
     temperature=15.0, top_k=40, top_p=0.9, target_model='gpt2', spm_vocab_size=150000, length=15,
     spm_model_name='spm_combinedAll_unigram_id', n_embd=512,
     model_name_or_path='../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model/gpt2/epoch_3-gpt2_id_wikiall_id.ckpt',
     vocab_model_dir='../temporary_before_move_to_git/id-pytorch-transformers/samples/wiki_datasets/trained_model')