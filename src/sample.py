from enumerator import SmilesEnumerator

sme = SmilesEnumerator()
smiles = ''
for i in range(10):

    print(sme.randomize_smiles("O[C@H]1[C@@H](O)[C@@H](O[C@@H]1COP(=O)(O)O)N2=CNc3c(S)ncnc23"))