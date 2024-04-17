# `optimum_interval_method`
NumPy implementation of the Optimum Interval Method from Yellin (2003).

## Install
I don't plan to publish this package to PyPI but you can still install it from source.
```
git clone https://github.com/tomeichlersmith/optimum_interval_method.git
pip install --user optimum_interval_method/
```

## Usage
You must have a locally-available table of maximum interval samples in order to
perform the Optimum Interval Method. This package allows for you to produce a table
to your desired precision and size and then cache that table for later use.
```python
import optimum_interval_method as oim
# define the table precision and size
#  more signal strengths and more trials require more space in memory to hold the table
#  below I show the default values which give /okay/ precision with a small table size
oim.new(
    max_signal_strenght = 20.0,
    n_test_mu = 100,
    n_trials = 1_000
)
```
Using the table and performing the OIM then means loading the table and asking it to
evaluate the maximum allowed signal strength given an input data set.
```python
o = oim.load()
o.max_signal_strength_allowed(data)
```
The input `data` array is required to have the _last_ axis be the one indexing the events.
These events should already be transformed into a uniformly distributed variable according
to the signal model being tested.
