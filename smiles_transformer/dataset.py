import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from enumerator import SmilesEnumerator
from utils import split

class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm = self.random_transform(sm)
        return split(sm)

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class STDataset(Dataset):

    def __init__(self, corpus_path, vocab, seq_len=203, transform=Randomizer(), is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train
        self.transform = transform
        df = pd.read_csv(corpus_path)
        self.data_size = len(df)
        self.firsts = df['first'].values
        self.seconds = df['second'].values

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        sm1, (sm2, is_same_label) = self.firsts[item], self.get_random_pair(item)
        sm1 = self.transform(sm1) # List
        sm2 = self.transform(sm2) # List
        masked_ids1, ans_ids1 = self.mask(sm1)
        masked_ids2, ans_ids2 = self.mask(sm2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        masked_ids1 = [self.vocab.sos_index] + masked_ids1 + [self.vocab.eos_index]
        masked_ids2 = masked_ids2 + [self.vocab.eos_index]

        ans_ids1 = [self.vocab.pad_index] + ans_ids1 + [self.vocab.pad_index]
        ans_ids2 = ans_ids2 + [self.vocab.pad_index]

        segment_embd = ([1]*len(masked_ids1) + [2]*len(masked_ids2))[:self.seq_len]
        bert_input = (masked_ids1 + masked_ids2)[:self.seq_len]
        bert_label = (ans_ids1 + ans_ids2)[:self.seq_len]

        padding = [self.vocab.pad_index]*(self.seq_len - len(bert_input))
        bert_input.extend(padding), bert_label.extend(padding), segment_embd.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_embd": segment_embd,
                  "is_same": is_same_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def get_random_pair(self, index):
        '''
        function: Find pair molecule. The boolean is_same_label is 1 
          for same and 0 for different molecules.
        '''
        rand = random.random()
        if rand<0.5: # Same molcule
            return self.firsts[index], 1
        else: # Different (but similar) molecule
            return self.seconds[index], 0

    def mask(self, sm):
        n_token = len(sm)
        masked_ids, ans_ids = [None]*n_token, [None]*n_token
        for i, token in enumerate(sm):
            if self.is_train: # Mask probablistically when training
                prob = random.random()
            else:  # Do not mask when predicting
                prob = 1.0

            if prob > 0.05:
                masked_ids[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                ans_ids[i] = 0
            else: # Mask
                prob /= 0.05
                # 80% randomly change token to mask token
                if prob < 0.8:
                    masked_ids[i] = self.vocab.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9:
                    masked_ids[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    masked_ids[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                ans_ids[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                
        return masked_ids, ans_ids


