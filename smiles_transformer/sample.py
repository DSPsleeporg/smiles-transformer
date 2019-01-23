from enumerator import SmilesEnumerator

sme = SmilesEnumerator()
smiles = ''
for i in range(10):

    print(sme.randomize_smiles("CN1C(=NS(=O)(=O)c2ccc(Cl)cc2)C(=NN=P(c3ccccc3)(c4ccccc4)c5ccccc5)c6ccccc16COc1ccc(Cl)cc1c2cc([nH]n2)C(=O)Nc3ccc(OC)nc3"))