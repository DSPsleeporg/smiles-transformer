from smiles_transformer.build_vocab import WordVocab

def test_vocab():
    with open('data/corpus.txt', "r", encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=None, min_freq=1)
    assert len(vocab.stoi)==70
    assert len(vocab.itos)==70
    assert vocab.stoi['luke']==14
    assert vocab.itos[13]=='leia'
    assert vocab.freqs['galaxy']==1
    
