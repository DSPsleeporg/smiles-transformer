import numpy as np
import pandas as pd 
from .enumerator import SmilesEnumerator

class Transform(object):
    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm = self.random_transform(sm)
        return self.split(sm)

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

    def split(self, sm):
        '''
        function: Split SMILES into words
        input: A SMILES
        output: A string with space between words
        '''
        arr = []
        i = 0
        while i < len(sm)-1:
            if not sm[i] in ['C', 'B']:
                arr.append(sm[i])
                i += 1
            elif sm[i]=='C' and sm[i+1]=='l':
                arr.append(sm[i:i+2])
                i += 2
            elif sm[i]=='B' and sm[i+1]=='r':
                arr.append(sm[i:i+2])
                i += 2
            else:
                arr.append(sm[i])
                i += 1
        if i == len(sm)-1:
            arr.append(sm[i])
        return ' '.join(arr) 


class DataLoder:

    def __init__(self, df, batch_size):
        self.batch_size = batch_size
        self.data_size = len(df)
        self.first = df['first'].values
        self.second = df['second'].values
        

    def __call__(self):
        firsts, seconds = self.sample_random_batch()
        firsts = self.random_transform(firsts)
        seconds = self.random_transform(seconds)
        # split
        # mask
        return firsts, seconds

    def sample_random_batch(self):
        '''
        function: Random transformation for SMILES. It may take some time.
        outputs: 
          firsts: A list of smiles
          seconds: A list of smiles. Same as firsts or not
          labels: A list of booleans indicating same or not
        '''
        rand = np.random.rand()
        rands = np.random.choice(self.data_size, self.batch_size, replace=False)
        firsts = self.first[rands]
        if rand<0.5: # Same molcules
            seconds = self.first[rands]
            labels = [1]*self.batch_size # Same
        else: # Different molecules
            seconds = self.second[rands]
            label = [0]*self.batch_size # Different
        return firsts, seconds, labels


