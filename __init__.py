"""numpy implementation of the optimum interval method

Yellin (2003) https://arxiv.org/abs/physics/0203002
"""


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
        pickle.dump(cache_file, oim)
    return oim


def load() -> _oim.OptimumIntervalMethod:
    """Load the table from the cache, throw exception if no cache exists"""

    if __cache_location.is_file():
        with open(__cache_location, 'rb') as cache_file:
            return pickle.load(cache_file)
    raise ValueError('No cache file to load OIM table from. Make sure to call `new` to define the test signal strengths and the size of the table.')
