''''
Run module to train tokenizer model and make vocabulary from pretrained dataset for Large Language Model from Power Law Decoder Representations
'''
import os
import logging
import time

import tensorflow_datasets as tfds
import datasets as hfds
import tensorflow as tf
import sentencepiece_tokenizer as spt


logging.getLogger('tensorflow').setLevel(logging.ERROR)

class sentencepiece_src_tokenizer:
    '''
    Creates a a custom sentencepiece tokenizer object and vocabulary from dataset.
    '''
    def __init__(self,
                 src_lang="en",
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_name='falcon-refinedweb',
                 data_source="hf",
                 split_style="index",
                 train_intvl=None,
                 src_spm_vocab_path="./refinedweb_pretrain_en_vocab.txt",
                 model_path = "./refinedweb_pretrain_en_tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 sp_tokenizer_params=None,
                 shuffle_files=False
                 ):
        '''
        Arguments:
            src_lang: Source language abbreviation as string. en by default.
            dataset_file: Location of dataset on disk to load.
            dataset_name: A name used for running a pre-defined preprocessing procedure for dataset.
            data_source: hf for loading a huggingface dataset or tf for loading a tensorflow dataset.
            split_style: index or percent to use as split train_intvl.
            train_intvl: A tuple for start and end indices/percent for dataset split. None loads all data.
            src_spm_vocab_path: Path to source vocabulary file to be created.
            model_path: Tokenizer model path location to save under or load from.
            load_tokenizer_model: If True loads tokenizer model at model_path path. 
                                  Dataset is not used to create vocabulary and tokenizer. Default is False.
            make_tokenizer: If True tokenizer is created from vocabulary and saved at model_path. 
                            If False only vocabulary is created from dataset. Default is True.
            sp_tokenizer_params: Parameter dict for sentencepiece tokenizer.
            shuffle_files: if True, shuffle dataset files.
        '''

        self.load_tokenizer_model=load_tokenizer_model
        self.model_path = model_path
        self.src_lang = src_lang
        self.make_tokenizer=make_tokenizer

        if self.load_tokenizer_model:
            #load tokenizer model only from model_path.
            print("TOKENIZER INITIALIZED FROM SAVED MODEL")
            self.tokenizers=tf.saved_model.load(self.model_path)
            print([item for item in dir(getattr(self.tokenizers, self.src_lang, None)) if not item.startswith('_')])
        else:
            #prepare dataset, create vocabularies, create tokenizers and save at model_path.
            self.src_spm_vocab_path = src_spm_vocab_path

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
        
            self.src_proto_path=self.sp_tokenizer_params["proto_output_file"]
            self.src_proto_path=f"{self.src_proto_path}.model"

            # create vocabulary generator objects
            self.src_vocab_obj = spt.gen_sp_proto(sp_tokenizer_params)
            if self.sp_tokenizer_params["data_as_text_file"] is None:
                print("LOADING DATASET")
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    if split_style=="percent":
                        if data_source=="tf":
                            #load only percentage of train data
                            examples, metadata = tfds.load(dataset_file,
                                                            split=[f"train[{start_ind}%:{end_ind}%]"],
                                                            with_info=True, as_supervised=False, shuffle_files=shuffle_files)
                        elif data_source=="hf":
                            examples = hfds.load_dataset(dataset_file,
                                                            split=[f'train[{start_ind}%:{end_ind}%]'])
                        else:
                            print(f"Warning: Invalid data source specified: {data_source}. Choose from tf or hf")
                    if split_style=="index":
                        if data_source=="tf":
                            #load only percentage of train data
                            examples, metadata = tfds.load(dataset_file,
                                                            split=[f"train[{start_ind}:{end_ind}]"],
                                                            with_info=True, as_supervised=False, shuffle_files=shuffle_files)
                        elif data_source=="hf":
                            examples = hfds.load_dataset(dataset_file,
                                                            split=[f'train[{start_ind}:{end_ind}]'])
                        else:
                            print(f"Warning: Invalid data source specified: {data_source}. Choose from tf or hf")
                    else:
                        print(f"Warning: Invalid split style specified: {split_style}. Choose from percent or index")
                else:
                    #load all data
                    if data_source=="tf":
                        examples, metadata = tfds.load(dataset_file, split=["train"], 
                                                        with_info=True, as_supervised=False, shuffle_files=shuffle_files)
                    elif data_source=="hf":
                            examples = hfds.load_dataset(dataset_file, split=["train"])
                    else:
                        print(f"Warning: Invalid data source specified: {data_source}. Choose from tf or hf")
                
                #do additional preprocessing to convert dataset to tf dataset format after loading
                if data_source=="hf":
                    if dataset_name in ['falcon-refinedweb']:
                        examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                    else:
                        examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                    #print a few samples from dataset
                    print(f"Printing a few examples from preprocessed dataset {dataset_name}")
                    for i in examples[0].take(3):
                        print(i)
                    else:
                        print(f"Warning: Invalid dataset name {dataset_name}. hf preprocessing not done.")

                self.train_examples = examples[0]
                if data_source=="tf":
                    self.metadata=metadata
            else:
                print(f"SKIPPED LOADING DATASET. DATASET AS TEXT FILE PROVIDED AT:{self.sp_tokenizer_params['data_as_text_file']}")

            print("TOKENIZER INITIALIZED FOR CREATION FROM VOCABULARY")
            self.tokenizers = tf.Module()

            print("MAKING VOCABULARIES")
            self.src_make_proto()
            print("VOCABULARIES DONE")

            if self.make_tokenizer:
                print("MAKING TOKENIZER")
                self.src_make_tokenizer()
                print("TOKENIZER DONE")

    def src_make_proto(self):
        '''
        Method to create sentencepiece proto file and vocabulary from dataset.
        Returns: path to the proto file
        '''

        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
            self.src_proto_path= None
        else:
            if self.src_proto_path:
                print(f"Creating  {self.src_lang} proto file for tokenizer at {self.src_proto_path}")
                start=time.time()
                if self.sp_tokenizer_params["data_as_text_file"] is None:
                    self.src_vocab_obj.generate_sp_proto(self.train_examples, vocab_filepath=self.src_spm_vocab_path)
                else:
                    self.src_vocab_obj.generate_sp_proto(None, vocab_filepath=self.src_spm_vocab_path)
                print(f"{self.src_lang} proto file done in {time.time() - start:.2f} s")

        return self.src_proto_path

    def src_make_tokenizer(self):
        '''
        Generate a custom tokenizer using the proto file and vocabulary from dataset.
        Returns:
            Tokenizer model and path to tokenizer model.
        '''

        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
        else:
            if os.path.isfile(os.path.abspath(self.src_proto_path)):
                lower_case=self.sp_tokenizer_params["lowercase"]
                print(f"Creating tokenizer for {self.src_lang} with lower case set as {lower_case}")
                start=time.time()
                setattr(self.tokenizers, self.src_lang, 
                        spt.CustomTokenizer(proto_file_path=self.src_proto_path, vocab_path=self.src_spm_vocab_path))
                print(f"en tokenizer done in {time.time()-start:.2f} s")
            else:
                print(f"No vocab file for {self.src_lang}, tokenizer not created")

            if self.model_path:
                print(f"Saving tokenizer as saved model at {os.path.abspath(self.model_path)}")
                tf.saved_model.save(self.tokenizers, os.path.abspath(self.model_path))

        return self.tokenizers, self.model_path
