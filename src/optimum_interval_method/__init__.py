"""numpy implementation of the optimum interval method

Yellin (2003) https://arxiv.org/abs/physics/0203002

You must have a locally-available table of maximum interval samples in order to
perform the Optimum Interval Method. This package allows for you to produce a table
to your desired precision and size and then cache that table for later use.

    import optimum_interval_method as oim
    # define the table precision and size
    #  more signal strengths and more trials require more space in memory to hold the table
    #  below I show the default values which give /okay/ precision with a small table size
    # load_or_new will check if the cache exists and only create a new table
    # if the cache does not exist, use new instead if you want to overwrite the cache
    oim.load_or_new(
        max_signal_strenght = 20.0,
        n_test_mu = 100,
        n_trials = 1_000
    )

Using the table and performing the OIM then means loading the table and asking it to
evaluate the maximum allowed signal strength given an input data set.

    oim.max_signal_strength_allowed(data)

The input `data` array is required to have the _last_ axis be the one indexing the events.
These events should already be transformed into a uniformly distributed variable according
to the signal model being tested.
"""


import pathlib
import pickle


import numpy as np


from . import _oim


__cache_location = (
  pathlib.Path(__file__).parent.resolve()
  / 'max_interval_size_cdf_cache.pkl'
)


def new(
    max_signal_strength = 20.0,
    n_test_mu = 100,
    n_trials = 1_000
) -> _oim.OptimumIntervalMethod:
    """Actively generate the table in memory, writing the newly
    generated table to the cache location

    Parameters
    ----------
    max_signal_strength: float
        maximum signal strength $\mu$ to have in the table
    n_test_mu: int
        number of signal strengths to have in table
    n_trials: int
        number of trials per signal strength to hold in the table

    Returns
    -------
    _oim.OptimumIntervalMethod
        object that implements the OIM with the generated table
    

    See Also
    --------
    _sample_generation.generate_max_interval_samples
        the function that implements the table generation
    """
    from . import _sample_generation
    mu_values = np.linspace(0.0, max_signal_strength, n_test_mu)
    table, mu, k, n = _sample_generation.generate_max_interval_samples(mu_values, n_trials)
    oim = _oim.OptimumIntervalMethod(
        table = table,
        mu_values = mu,
        k_values = k,
        n_trials_per_mu = n
    )
    with open(__cache_location, 'wb') as cache_file:
        pickle.dump(oim, cache_file)
    return oim


def load() -> _oim.OptimumIntervalMethod:
    """Load the table from the cache, throw exception if no cache exists"""

    if __cache_location.is_file():
        with open(__cache_location, 'rb') as cache_file:
            return pickle.load(cache_file)
    raise ValueError('No cache file to load OIM table from. Make sure to call `new` to define the test signal strengths and the size of the table.')


def load_or_new(**kwargs) -> _oim.OptimumIntervalMethod:
    """Attempt to load the table from cache, choosing to create a new one with
    the input keyword arguments if no cache is available."""

    try:
        return load()
    except ValueError:
        return new(**kwargs)


def max_signal_strenght_allowed(*args, **kwargs):
    """Usability function to hold the loaded table for the user

    The user is required to already have a table in cache, so this
    will error out if no table can be loaded from cache.

    All arguments (both position and keyword) are passed to the
    _oim.OptimumIntervalMethod.max_signal_strength_allowed function.
    """
    
    if getattr(max_signal_strength_allowed, '_oim_calculator', None) is None:
        max_signal_strength_allowed._oim_calculator = load()
    return getattr(
        max_signal_strength_allowed,
        '_oim_calculator'
    ).max_signal_strength_allowed(*args, **kwargs)
