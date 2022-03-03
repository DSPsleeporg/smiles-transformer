# SMILES Transformer

[SMILES Transformer](http://arxiv.org/abs/1911.04738) extracts molecular fingerprints from string representations of chemical molecules.  
The transformer learns latent representation that is useful for various downstream tasks through autoencoding task.

## Requirement
This project requires the following libraries.

- NumPy
- Pandas
- PyTorch > 1.2
- tqdm
- RDKit

## Dataset
Canonical SMILES of 1.7 million molecules that have no more than 100 characters from Chembl24 dataset were used.  
These canonical SMILES were transformed randomly every epoch with [SMILES-enumeration](https://github.com/EBjerrum/SMILES-enumeration) by E. J. Bjerrum.  

## Pre-training
After preparing the SMILES corpus for pre-training, run:

```
$ python pretrain_trfm.py
```

Pre-trained model is [here](https://drive.google.com/file/d/1LwE2BzvtDaPGYv0OR6iBjmsqoloH885N/view?usp=sharing).

## Downstream Tasks
See `experiments/` for the example codes.

## Cite
```
@article{honda2019smiles,
    title={SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery},
    author={Shion Honda and Shoi Shi and Hiroki R. Ueda},
    year={2019},
    eprint={1911.04738},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
