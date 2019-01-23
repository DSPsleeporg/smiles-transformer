import numpy as np
from tqdm import tqdm

def sample(path, rands):
    with open(path + 'sentence_train.txt') as f:
        lines = f.readlines()
    
    with open(path + 'sentence_train_small.txt', 'a') as f:
        for i in tqdm(rands):
            sublines = lines[50*i:50*(i+1)]
            for line in sublines:
                f.write(line)

def main():
    rands = np.random.choice(1722298, 400000, replace=False)
    print('Making source file')
    sample('../data/Enum2Enum/source/', rands)
    print('Making target file')
    sample('../data/Enum2Enum/target/', rands)

if __name__ == '__main__':
    main()