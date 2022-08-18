# Phenix Fragments
An experimental toolbox for working with molecular fragments and generating chemical restraints for the Phenix project


# Installation
Install dependencies
```bash
git clone git@github.com:cschlick/phenix_fragments.git
cd phenix_fragments
conda env create --file=environment.yml
```

Installing dependencies with conda can be slow, perhaps try with mamba:
```
conda install -c conda-forge mamba
mamba env create --file=environment.yml
```

The above steps will create an environment names "frag". To remove the environment:
```bash
conda env remove -n frag
```

To install this package:
```bash
pip install .
```

To reinstall just this package after making changes:
```bash
pip install . --force-reinstall
```
