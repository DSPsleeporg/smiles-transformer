import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import time
import argparse
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def get_fps(candidates):
    smiles, fps = [], []
    for sm in tqdm(candidates):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        smiles.append(sm)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) # ECFP4
        fps.append(fp)
    return smiles, fps

def choose_pair(smiles, fps, n,m):
    N = len(smiles)
    for i in tqdm(range((m*N)//n, ((m+1)*N//n))):
        tanimotos = np.array(DataStructs.BulkTanimotoSimilarity(fps[i],fps)) # Tanimoto similarity
        tanimotos[i] = 0 # Choose second largest (not the self)
        arg_i = np.argmax(tanimotos)
        with open('../data/chembl_24_bert.csv', 'a') as f:
            f.write('%s,%s,,,%f\n' %(smiles[i],smiles[arg_i],tanimotos[arg_i]))

def main():
    parser = argparse.ArgumentParser(description='Make pairs of similar molecules')
    parser.add_argument('--n_process', '-p', type=int, default=1, help='number of processes paralelled')
    parser.add_argument('--file_path', '-f', type=str, default='../data/chembl24_train.csv', help='specify which file to preprocess')
    args = parser.parse_args()
    print('%d processes paralelled.' % args.n_process)
    print('File path: %s' % args.file_path )
    print('Start preprocessing')

    n_process = args.n_process   
    df = pd.read_csv(args.file_path)
    candidates = df['canonical_smiles'].values
    smiles, fps = get_fps(candidates[:10000])
    N = len(smiles)
    print('N=%d'%N)
    
    with open('../data/chembl_24_bert.csv', 'a') as f:
        f.write('first,second,first_sp,second_sp,tanimoto\n')
    # Execute multiprocessed choose_pair
    start = time.time()
    processes = []
    for i in range(n_process):
        process = multiprocessing.Process(target=choose_pair, args=([smiles, fps, n_process,i]))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    elapsed = time.time()-start
    print('%f seconds elapsed.' %elapsed)

if __name__=='__main__':
    main()