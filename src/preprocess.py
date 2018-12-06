from tqdm import tqdm

for file_idx in range(172):
    with open('../data/Enum2Enum/'+format(file_idx,'03d')+'.txt') as f:
        lines = f.readlines()
    if file_idx==0:
        with open('../data/Enum2Enum/sentence_test.txt', 'a') as f:
            for smiles in tqdm(lines):
                new_smiles = ' '.join(list(smiles[:-2]))
                f.write(new_smiles+'\n')
    else:
        with open('../data/Enum2Enum/sentence_test.txt', 'a') as f:
            for smiles in tqdm(lines):
                new_smiles = ' '.join(list(smiles[:-2]))
                f.write(new_smiles+'\n')