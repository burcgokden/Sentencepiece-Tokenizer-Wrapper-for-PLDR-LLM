'''
Module for creating a sentencepiece tokenizer model and vocabulary and define a custom tokenizer class as wrapper
for the tokenizer model.
'''

import os
import pathlib

import numpy as np
import random
import tensorflow as tf
import keras_nlp
import sentencepiece as spm
import time

# # Uncomment for reproducibility.
# print("Setting random seeds")
# random.seed(1234)
# np.random.seed(1234)
# tf.random.set_seed(1234)
# spm.set_random_generator_seed(1234)

tf.get_logger().setLevel('ERROR')

class gen_sp_proto():
    '''
    This class is a wrapper for generating sentencepiece model and vocabulary.
    '''
    def __init__(self, sp_tokenizer_params=None):
        '''
        Arguments:
            sp_tokenizer_params: dictionary of parameters for sentencepiece tokenizer model trainer
        '''


        if sp_tokenizer_params:
            self.sp_tokenizer_params= sp_tokenizer_params
        else:
            self.sp_tokenizer_params={"lowercase": False, 
                                      "vocabulary_size": 32000, 
                                      "model_type": "unigram", 
                                      "proto_output_file": "my_sp_model",
                                      "num_threads": None,
                                      "input_sentence_size": None,
                                      "max_sentence_length":None,
                                      "shuffle_input_sentence": True,
                                      "minloglevel": None,
                                      "use_iterator": False,
                                      "data_as_text_file": None
                                      }
            

        self.reserved_tokens= ["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]", "[CLS]"]

    def generate_sp_proto(self, ds, vocab_filepath=None):
        '''
        Generates vocabulary from tensorflow dataset
        Arguments:
            ds: A tensorflow dataset containing text.
            vocab_filepath: a file path to save vocabulary as text file.
        Returns:
            A vocabulary and saves text file containing vocabulary from dataset.
        '''

        proto_output_file=self.sp_tokenizer_params["proto_output_file"]
        vocabulary_size=self.sp_tokenizer_params["vocabulary_size"]
        model_type=self.sp_tokenizer_params["model_type"]
        lowercase=self.sp_tokenizer_params["lowercase"]
        num_threads=self.sp_tokenizer_params["num_threads"]
        input_sentence_size=self.sp_tokenizer_params["input_sentence_size"]
        max_sentence_length=self.sp_tokenizer_params["max_sentence_length"]
        shuffle_input_sentence=self.sp_tokenizer_params["shuffle_input_sentence"]
        minloglevel=self.sp_tokenizer_params["minloglevel"]
        use_iterator=self.sp_tokenizer_params["use_iterator"]
        data_as_text_file=self.sp_tokenizer_params["data_as_text_file"]

        if use_iterator:
            if proto_output_file:
                model_writer=open(f"{proto_output_file}.model", "wb")
            else:
                print("Please specify a file path to write the proto output file.")

            spm.SentencePieceTrainer.train(
                sentence_iterator=ds.as_numpy_iterator(),
                model_writer=model_writer,
                vocab_size=vocabulary_size,
                model_type=model_type,
                normalization_rule_name="nmt_nfkc_cf" if lowercase else "nmt_nfkc",
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                user_defined_symbols=["[SEP]", "[CLS]"],
                split_digits=True,
                byte_fallback=True,
                hard_vocab_limit=False,
                unk_piece="[UNK]",
                bos_piece="[START]",
                eos_piece="[END]",
                pad_piece="[PAD]",
                max_sentence_length=int(max_sentence_length) if max_sentence_length is not None else 4192, 
                num_threads=num_threads if num_threads is not None else 16,
                input_sentence_size=input_sentence_size if input_sentence_size is not None else 0,
                shuffle_input_sentence=shuffle_input_sentence if shuffle_input_sentence is not None else True,
                minloglevel=minloglevel if minloglevel is not None else 0
            )
        else:
            #write ds to text file line by line.
            if data_as_text_file is None:
                print("WRITING DATASET TO TEXT FILE")
                start=time.time()
                with open(f"{proto_output_file}-data-raw.txt", "w") as dstxtfile:
                    for t in ds.as_numpy_iterator():
                        t=t.decode('utf-8')
                        dstxtfile.write(t)
                #remove blank lines between sentences and write to a final file.
                with open(f"{proto_output_file}-data-raw.txt", "r"
                          ) as dsreadfile, open(f"{proto_output_file}-data.txt", "w") as dswritefile:
                    for line in dsreadfile:
                        if line.strip():
                            dswritefile.write(line)
                    
                data_as_text_file=f"{proto_output_file}-data.txt"
                print(f"WRITING DATASET TO TEXT FILE FINISHED IN {time.time()-start:.2f}s AT: {data_as_text_file}")
            else:
                print(f"USING PROVIDED DATA TEXT FILE AS INPUT: {data_as_text_file}")
            
            normalization_rule_name="nmt_nfkc_cf" if lowercase else "nmt_nfkc"
            max_sentence_length=int(max_sentence_length) if max_sentence_length is not None else 4192
            num_threads=num_threads if num_threads is not None else 16
            input_sentence_size=input_sentence_size if input_sentence_size is not None else 0
            shuffle_input_sentence=shuffle_input_sentence if shuffle_input_sentence is not None else True 
            minloglevel=minloglevel if minloglevel is not None else 0
            
            #define the command line string
            sp_cmd_line=f"--input={data_as_text_file} --model_type={model_type} --model_prefix={proto_output_file} --vocab_size={vocabulary_size} \
                          --normalization_rule_name={normalization_rule_name} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
                          --user_defined_symbols=[SEP],[CLS] --split_digits=True --byte_fallback=True --hard_vocab_limit=False \
                          --unk_piece=[UNK] --bos_piece=[START] --eos_piece=[END] --pad_piece=[PAD] --max_sentence_length={max_sentence_length} \
                          --num_threads={num_threads} --input_sentence_size={input_sentence_size} --shuffle_input_sentence={shuffle_input_sentence} \
                          --minloglevel={minloglevel}"
            print(sp_cmd_line)

            #give some time for interfaces to post outputs
            print("STARTING SENTENCEPIECE TRAINING")
            time.sleep(5)
 
            spm.SentencePieceTrainer.train(sp_cmd_line)

        #write the vocabulary to a text file.
        if vocab_filepath:
            print(f"writing vocabulary to file: {os.path.abspath(vocab_filepath)}")
            self.write_vocab_file(os.path.abspath(vocab_filepath))
        else:
            print("Skipped writing vocabulary to file")

        return f"{proto_output_file}.model"

    def write_vocab_file(self, vocab_filepath):
        '''
        Write vocabulary to file one token at a time.
        Arguments:
            vocab_filepath: text file path for vocabulary.
        Returns:
            A text file containing vocabulary.
        '''

        proto_output_file=self.sp_tokenizer_params["proto_output_file"]
        spm_tokenizer=keras_nlp.tokenizers.SentencePieceTokenizer(proto=f"{proto_output_file}.model",
                                                    sequence_length=None,
                                                    dtype="int64"
                                                    )
        
        vocab=spm_tokenizer.get_vocabulary()
        with open(os.path.abspath(vocab_filepath), 'w') as f:
            for token in vocab:
                print(token, file=f)


class CustomTokenizer(tf.Module):
    '''
    A custom tokenizer module class wrapper modified from subword tutorial at
    https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer
    https://github.com/tensorflow/text/blob/master/docs/guide/subwords_tokenizer.ipynb
    '''

    def __init__(self, proto_file_path, vocab_path, **kwargs):

        super().__init__(**kwargs)
        self.tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=proto_file_path,
                                                    sequence_length=None,
                                                    dtype="int32"
                                                    )
        # self._reserved_tokens = reserved_tokens
        self._reserved_tokens= ["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]", "[CLS]"]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shape [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        enc=tf.cast(enc, dtype=tf.int64)
        enc = self.add_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        tokenized=tf.cast(tokenized, dtype=tf.int32)
        words = self.tokenizer.detokenize(tokenized)
        return words

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
    
    def add_end(self, ragged):
        END = tf.argmax(tf.constant(self._reserved_tokens) == "[END]", output_type=tf.int64)
        count = ragged.bounding_shape()[0]
        ends = tf.fill([count, 1], END)
        return tf.concat([ragged, ends], axis=1)
