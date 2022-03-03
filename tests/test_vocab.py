import os
from smiles_transformer.build_vocab import WordVocab

def test_vocab():
    cwd = os.getcwd() 
    if cwd[-5:]=='tests':
        path = 'data/corpus.txt'
    else:
        path = 'tests/data/corpus.txt'
    with open(path, "r", encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=None, min_freq=1)
    assert len(vocab.stoi)==70
    assert len(vocab.itos)==70
    assert vocab.stoi['luke']==14
    assert vocab.itos[13]=='leia'
    assert vocab.freqs['galaxy']==1
    
