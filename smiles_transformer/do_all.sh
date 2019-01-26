#!/bin/sh

python preprocess_for_bert.py -p 32
python build_corpus.py
python build_vocab.py
python pretrain.py --gpu 0  -b 64 --train_data data/chembl24_bert_train_small.csv  --test_data data/chembl24_bert_test_small.csv 