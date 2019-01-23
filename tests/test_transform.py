
from smiles_transformer.dataloader import Transform

def test_split():
    transform = Transform()
    sm = 'C(=O)CC(Br)C[N+]CN'
    pred = 'C ( = O ) C C ( Br ) C [ N + ] C N'
    assert transform.split(sm)==pred