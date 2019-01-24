import argparse
import pandas as pd
from tqdm import tqdm

from utils import split

def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')
    parser.add_argument('--in_path', '-i', type=str, default='data/chembl24_bert_train.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default='data/chembl24_corpus.txt', help='output file')
    args = parser.parse_args()

    smiles = pd.read_csv(args.in_path)['first'].values
    with open(args.out_path, 'a') as f:
        for sm in tqdm(smiles):
            f.write(split(sm)+'\n')
    print('Built a corpus file!')

if __name__=='__main__':
    main()



