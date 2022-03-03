import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Convert smi file to csv file')
    parser.add_argument('--in_path', '-i', type=str, default='data/GDB/GDB08.smi', help='input file')
    parser.add_argument('--out_path_1', '-o1', type=str, default='data/gdb08_bert_train.csv', help='output file (train)')
    parser.add_argument('--out_path_2', '-o2', type=str, default='data/gdb08_bert_test.csv', help='output file (test)')
    parser.add_argument('--max', '-m', type=int, default=2e6, help='Maximum number of molecules')
    args = parser.parse_args()
    print('Input file: {}'.format(args.in_path))
    print('Start preprocessing')

    smiles = []
    with open(args.in_path) as f:
        lines = f.readlines()
    for l in lines:
        smiles.append(l.replace('\n', ''))
    del lines
    smiles = np.array(smiles)
    N = len(smiles)
    print('The dataset contains {} molecules'.format(N))
    
    rands = np.random.choice(N, min(N,args.max), replace=False)
    smiles_train = smiles[rands[:N//2]]
    df_train = pd.DataFrame(data=smiles_train, columns=['canonical_smiles'])
    df_train.to_csv(args.out_path_1, index=False)
    del smiles_train, df_train
    smiles_test = smiles[rands[N//2:]]
    df_test = pd.DataFrame(data=smiles_test, columns=['canonical_smiles'])
    df_test.to_csv(args.out_path_2, index=False)
    print('Each set contains {} molecules'.format(N//2))

if __name__=='__main__':
    main()