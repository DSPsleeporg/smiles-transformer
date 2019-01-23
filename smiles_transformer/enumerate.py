import pandas as pd
from tqdm import tqdm
from enumerator import SmilesEnumerator

print('Loading Canonical SMILES data...')
df = pd.read_csv('../data/chembl_24.csv')
cans = df['canonical_smiles'].values
#cans = cans[111*10000:]
del df

sme = SmilesEnumerator()
enum_times = 50
print('Started enumerating for {} times'.format(enum_times))

for i,can in enumerate(tqdm(cans)):
    with open('../data/Enum2Enum/target/'+format(i//10000,'03d')+'.txt', 'a') as f:
        for j in range(enum_times):
            enum = sme.randomize_smiles(can)
            if enum is None:
                break
            f.write(enum + '\n')
print('Enumeration done.')
