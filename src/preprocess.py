from tqdm import tqdm

def insert_space(path):
    for file_idx in tqdm(range(172)):
        with open(path + format(file_idx,'03d')+'.txt') as f:
            lines = f.readlines()
        if file_idx==0:
            with open(path + 'sentence_test.txt', 'a') as f:
                for smiles in lines:
                    new_smiles = ' '.join(list(smiles[:-2]))
                    f.write(new_smiles+'\n')
        else:
            with open(path + 'sentence_train.txt', 'a') as f:
                for smiles in lines:
                    new_smiles = ' '.join(list(smiles[:-2]))
                    f.write(new_smiles+'\n')

def main():
    print('Making source file')
    insert_space('../data/Enum2Enum/source/')
    print('Making target file')
    insert_space('../data/Enum2Enum/target/')

if __name__ == '__main__':
    main()