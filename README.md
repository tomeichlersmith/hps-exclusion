# HPS Exclusion
Helper modules for performing exclusions with HPS.

### `optimum_interval_method`
NumPy implementation of the Optimum Interval Method from Yellin (2003).

### `production`
Functional implementation for calculating the total signal production
for a variety of dark photon masses.

### `models`
Implementation of different models functions for calculating decay rates
and branching ratios.

## Usage
I don't plan to publish this package to PyPI and I don't think it is at a stage where
it can be installed. Please just clone this directory into your workspace and load
it from source.
For easier bug reporting, I would request that you choose a tag so that it is
obvious what source code you are using.
```
git clone --branch <tag> https://github.com/tomeichlersmith/hps-exclusion.git exclusion
pip install -r exclusion/requirements.txt
```
If you find yourself editing these files while using them within a notebook, the `autoreload`
extension is helpful for making the modules reload if the source files have changed.
```
%load_ext autoreload
%autoreload 2
```
