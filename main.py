import sys
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import tokenization

folder = 'multi_cased_L-12_H-768_A-12'

config_path = folder+'/bert_config.json'
checkpoint_path = folder+'/bert_model.ckpt'
vocab_path = folder+'/vocab.txt'

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model.summary()