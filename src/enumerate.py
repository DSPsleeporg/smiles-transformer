import pandas as pd
from tqdm import tqdm
from enumerator import SmilesEnumerator

print('Loading Canonical SMILES data...')
df = pd.read_csv('../data/chembl_24.csv')
cans = df['canonical_smiles'].values
cans = cans[58*10000:]
del df

sme = SmilesEnumerator()
enum_times = 50
print('Started enumerating for {} times'.format(enum_times))

for i,can in enumerate(tqdm(cans)):
    with open('../data/Enum2Enum/'+format(i//10000+58,'03d')+'.txt', 'a') as f:
        for j in range(enum_times):
            f.write(sme.randomize_smiles(can) + '\n')
print('Enumeration done.')
