# SMILES Transformer

SMILES Transformer extracts beautiful molecular fingerprints by BERT-like pretraining.

## Requirement

This project requires additional libraries.

- numpy
- pandas
- PyTorch: 1
- tqdm
- rdkit

```
$ pip install tqdm
```

## Dataset
Canonical SMILES of 1.7 million molecules that have no more than 100 characters from Chembl24 dataset were used.  
These canonical SMILES were transformed randomly every epoch with [SMILES-enumeration](https://github.com/EBjerrum/SMILES-enumeration) by E. J. Bjerrum.  
