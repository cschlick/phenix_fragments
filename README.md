# Phenix Fragments
An experimental toolbox for working with molecular fragments and generating chemical restraints for the Phenix project


# Installation
Clone the repo 
```bash
git clone git@github.com:cschlick/phenix_fragments.git
cd phenix_fragments
```
Install dependencies using conda via the mamba. To install mamba:
```bash
conda install -c conda-forge mamba

```
If a fresh environment is desired:
```bash
mamba env create -n frag --file=environment.yml
```
Or, to add dependencies to an active environment:
```bash
mamba env update --file=environment.yml
```

To install this package:
```bash
pip install . --force-reinstall
```
