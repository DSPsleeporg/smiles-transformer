from smiles_transformer.dataset import Randomizer

def test_split():
    randomizer = Randomizer()
    sm = 'C(=O)CC(Br)C[N+]CN'
    pred = 'C ( = O ) C C ( Br ) C [ N + ] C N'
    assert randomizer.split(sm)==pred