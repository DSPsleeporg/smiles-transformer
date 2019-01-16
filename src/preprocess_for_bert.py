import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import time
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
    if m<n-1:
        lst = smiles[m*(N//n):(m+1)*(N//n)]
    else:
        lst = smiles[m*(N//n):]
    for i,sm in tqdm(enumerate(lst)):
        tanimotos = np.array(DataStructs.BulkTanimotoSimilarity(fps[i],fps)) # Tanimoto similarity
        idx = np.argpartition(tanimotos,-2)[-2] # Choose second largest (not the self)
        #idx = tanimotos.argmax()
        with open('../data/chembl_24_bert.csv', 'a') as f:
            f.write('%s,%s,,,%f\n' %(sm,smiles[idx],tanimotos[idx]))

def main():
    num_process = 16
    
    df = pd.read_csv('../data/chembl_24.csv')
    candidates = df['canonical_smiles'].values
    smiles, fps = get_fps(candidates)
    N = len(smiles)
    print('N=%d'%N)
    
    with open('../data/chembl_24_bert.csv', 'a') as f:
        f.write('first,second,first_sp,second_sp,tanimoto\n')
    # Execute multiprocessed choose_pair
    start = time.time()
    processes = []
    for i in range(num_process):
        process = multiprocessing.Process(target=choose_pair, args=([smiles, fps, num_process,i]))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    elapsed = time.time()-start
    print('%f seconds elapsed.' %elapsed)

if __name__=='__main__':
    main()