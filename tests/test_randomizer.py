from smiles_transformer.utils import split

def test_split():
    sm = 'C(=O)CC(Br)C[N+]CN'
    pred = 'C ( = O ) C C ( Br ) C [ N + ] C N'
    assert split(sm)==pred