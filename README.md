## Sentencepiece Tokenizer Wrapper for PLDR-LLM

This repository implements a wrapper code for generating a Sentencepiece Vocabulary and Tokenizer model from RefinedWeb dataset using tensorflow-text package. The tokenizers generated with this wrapper script are used in the research article: [PLDR-LLM: Large Language Model From Power Law Decoder Representations](https://arxiv.org/abs/2410.16703).

More information on Sentencepiece tokenizer can be found in articles:
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) 
- [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959). 

The Git repo for Sentencepiece module is at [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece). A tutorial for creating custom subword tokenizer class using tensorflow can be found at [https://www.tensorflow.org/text/guide/subwords_tokenizer](https://www.tensorflow.org/text/guide/subwords_tokenizer).

#### Key features

- Builds a Sentencepiece tokenizer model from dataset and wraps in a custom tensorflow tokenizer class for PLDR-LLM training and evaluation.
- Optimized for preprocessing RefinedWeb dataset to generate a tokenizer model.
- The tokenizer implements reserved tokens "[PAD]", "[UNK]", "[START]", "[END]", "[SEP]", "[CLS]" into vocabulary for research purposes.
- The custom tokenizer class adds an "[END]" token at the end of each sentence and supports "[PAD]" token for padding.

#### Setting Hyperparameters for Sentencepiece Model Training:
Some of the hyperparameters for Sentencepiece can be provided through a parameter dictionary. proto_name is used for creating file names for sentencepiece model, tokenizer and preprocessed dataset files. If preprocessed text file is available, its path can be provided through "data_as_text_file" key.

```python
import pretrain_make_sentencepiece_tokenizer as ptmspt

proto_name=os.path.abspath("/file/path/to/tokenizer/model")
sp_tokenizer_params={"lowercase":False, 
                     "vocabulary_size":32000, 
                     "model_type":"unigram", 
                     "proto_output_file":proto_name, 
                     "num_threads":None,
                     "input_sentence_size":5000000,
                     "max_sentence_length":int(4192*30),
                     "shuffle_input_sentence":True,
                     "minloglevel":0,
                     "use_iterator":False,
                     "data_as_text_file":None
                    }
```

Below features of the tokenizer model are predefined in the wrapper module:
```
--pad_id=0 --unk_id=1 
--bos_id=2 --eos_id=3
--user_defined_symbols=[SEP],[CLS] 
--split_digits=True 
--byte_fallback=True 
--hard_vocab_limit=False
--unk_piece=[UNK] --bos_piece=[START] 
--eos_piece=[END] --pad_piece=[PAD]
```

#### Training a Sentencepiece Model and Tokenizer:

The tokenizer and a sentencepiece model can be trained as follows.

```python
refinedweb_sp_tokenizer = ptmspt.sentencepiece_src_tokenizer(
                 src_lang='en',
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_name="falcon-refinedweb",
                 data_source="hf",
                 split_style="index",
                 train_intvl=[0, 2000000],
                 src_spm_vocab_path=f"{proto_name}-vocab.txt",
                 model_path = f"{proto_name}-tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 sp_tokenizer_params=sp_tokenizer_params,
                 shuffle_files=True
                 )
```
This example also saves the tokenizer as as a tensorflow saved model in the same folder where sentencepiece proto model is saved.

#### Loading The Tokenizer

```python
import tensorflow as tf

model_name = "/path/to/tokenizer/model"

#load tokenizer model
sp_tokenizers = tf.saved_model.load(model_name)

#print items ins the tokenizers class
print([item for item in dir(sp_tokenizers) if not item.startswith('_')])

#assign "en" attribute to access tokenizer
sp_tokenizers_src=getattr(sp_tokenizers, "en", None)
```

#### Encoding and Decoding with The Tokenizer

Below methods are accessible through the tokenizer to encode text and decode tokens, view reserved tokens or get the vocabulary size and location.

```python
print(sp_tokenizers_src.tokenize(["Hello World!", "How do you do?"]))
print(sp_tokenizers_src.get_vocab_size())
print(sp_tokenizers_src.get_vocab_path())
print(sp_tokenizers_src.get_reserved_tokens())
print(sp_tokenizers_src.lookup(tf.convert_to_tensor([[4812, 979, 316, 3, 0], [592, 291, 282, 319, 3]], dtype=tf.int64)))
print(sp_tokenizers_src.detokenize(tf.convert_to_tensor([[4812, 979, 316, 3, 0], [592, 291, 282, 319, 3]], dtype=tf.int64)))
```
