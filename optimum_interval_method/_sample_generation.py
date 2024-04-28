"""module holding generation of max interval size samples"""

import numpy as np


def generate_max_interval_samples(test_mu, n_trials_per_mu):
    """Generate the table of trials given the np.array of signal strengths to test
    and the number of trials to MC sample per signal strength

    Parameters
    ----------
    test_mu: np.array, 1D
        the signal strength values we should test
    n_trials_per_mu: int
        number of trials to test with

    Returns
    -------
    tuple(3D np.array, 1D np.array, 1D np.array, int)
        the table of trials whose indices are (i_mu, k, i_trial)
        signal strengths represented indexed by (i_mu)
        k values represented indexed by (k)
        number of trials for each mu
    """

    # first sample the mu into number of events in each trial
    # trial_counts[mu_index, trial_index]
    trial_counts = np.swapaxes(
        np.random.poisson(
            test_mu,
            size=(n_trials_per_mu,*test_mu.shape)
        ),
        0,
        1
    )

    # generate uniformly random numbers for each trial in each mu
    # us[event_index, mu_index, trial_index]
    us = np.random.random(
        size=(np.max(trial_counts), *test_mu.shape, n_trials_per_mu)
    )
    # this over-generates numbers so we set any entries whose index
    # is greater than the number of counts equal to the signal value np.nan
    #  (we use np.nan since any arithmetic with np.nan produces np.nan)
    us[np.greater.outer(np.arange(us.shape[0]), trial_counts)] = np.nan
    # sort the entries in each event along the event index axis
    us = np.sort(us, axis=0)
    # add the edges of the distribution for calculating the interval sizes
    # uswe[event_index, mu_index, trial_index]
    uswe = np.full(
        (us.shape[0]+2, *us.shape[1:]),
        np.nan
    )
    # set the contents within the edges to the random samples from before
    uswe[1:-1,...] = us
    # set the lower edge to be zero
    uswe[0,...] = 0.0
    # set the upper edge to 1.0
    #   the index of the upper edge is found by checking where the event index
    #   is equal to the number of entries in the trial (+1 since we added the lower edge)
    uswe[np.equal.outer(np.arange(uswe.shape[0]), trial_counts+1)] = 1.0

    max_possible_k = np.max(trial_counts)+1
    k_values = np.arange(max_possible_k)
    # calculate the maximum interval containing k events for each mu and each trial
    # max_interval_by_k[k_index, mu_index, trial_index]
    max_interval_by_k = np.full(
        (max_possible_k, *uswe.shape[1:]),
        np.nan
    )
    # this for loop is not expected to be a performance bottleneck since we expect the
    # k to be limited to ~50
    # we use np.nan to ignore all of the signal values we inserted above
    for k in k_values:
        # calculate interval differences with k entries in them
        interval_differences = uswe[(k+1):,...]-uswe[:-(k+1),...]
        # replace any NaNs leftover with -1.0 so they are eliminated by the np.max
        interval_differences[np.isnan(interval_differences)] = -1.0
        # find maximum interval over the event index
        max_interval_by_k[k,...] = np.max(interval_differences, axis=0)

    # if a maximum interval is -1.0 then that means there was no interval for that k
    # for our purposes, we re-cast those values to a maximum interval of 1.0 since that
    # effectively means there is no way to get an interval of any size for that k
    max_interval_by_k[max_interval_by_k < 0.0] = 1.0

    # swap k and mu indices in table for easier access later on
    return np.swapaxes(max_interval_by_k, 0, 1), test_mu, k_values, n_trials_per_mu
