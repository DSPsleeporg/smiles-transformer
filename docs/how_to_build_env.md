# How to Build an Environment for Cheminformatics

## Install Anaconda 5.3
Download shell file to install Anaconda 5.3.1.  
https://www.anaconda.com/download/#linux  
Installed Python 3.7 and VS Code.

Add path to Anaconda.

```
$ echo 'export PATH=/home/user/anaconda3/bin:$PATH' \
>> ~/.bashrc
```

## Install basic libraries

```
$ pip install --upgrade pip
$ pip install tqdm nltk progressbar selenium
```

TensorFlow hasn't yet supported Python 3.7...

```
$ conda install python=3.6
$ pip install opencv-python tensorflow-gpu \
six cupy-cuda91 chainer
```

### Check installation of Chainer

```python
>>> import chainer
>>> chainer.backends.cuda.available
True
>>> chainer.backends.cuda.cudnn_enabled
True
```

## RDKit from conda

```
$ conda install -c rdkit rdkit
```

## DeepChem from source

DeepChem seems to support only CUDA 9.0...  
Install without GPU support.

```
$ pip install boost joblib mdtraj simdna
$ git clone \
 https://github.com/deepchem/deepchem.git
$ cd deepchem
$ bash scripts/install_deepchem_conda.sh deepchem
$ source activate deepchem
$ python setup.py install
$ nosetests -a '!slow' -v deepchem --nologcapture  
```

### How to use DeepChem

```
$ source activate deepchem
```
