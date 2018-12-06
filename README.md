# SMILES Transformer

SMILES Transformer extracts beautiful molecular fingerprints utilizing NLP and graph algorithms.

## Requirement

This project requires additional libraries.

- [NLTK](https://www.nltk.org/).
- progressbar

```
$ pip install nltk progressbar
```

## Dataset
Canonical SMILES of 1.7 million molecules that have no more than 100 characters from Chembl24 dataset were used.  
These canonical SMILES were enumerated 50 times with [SMILES-enumerator](https://github.com/EBjerrum/SMILES-enumeration) by E. Bjerrum.  
