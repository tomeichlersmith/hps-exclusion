"""module for generating, saving, and loading the max interval size LUT"""

from dataclasses import dataclass
import pathlib
import pickle

import numpy as np


def _generate_max_interval_samples(test_mu, n_trials_per_mu):
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


def _largest_intervals_by_k(data):
    """generate list of largest intervals containing k data points according to the input data

    The returned array has one additional axis in the zero'th position
    which is indexed by $k$ (the number of events within the interval).

    Parameters
    ----------
    data: np.array N-D
        input array of data, it can be any dimension but the last index needs to
        be the index of data points. We expect the data points to already be transformed
        into a uniform distribution between 0 and 1 and be sorted.

    Returns
    -------
    np.array, (N+1)-D
        The index of this array is the same as the input data array but with the
        index over k in the zero'th position.
    """
    # assume the last axis is the one for the data of the events
    data_with_edges = np.full((*data.shape[:-1], data.shape[-1]+2), 0.)
    data_with_edges[...,1:-1] = data
    data_with_edges[...,-1] = 1.
    return np.array([
        np.max(data_with_edges[...,(k+1):]-data_with_edges[...,:-1*(k+1)], axis=-1)
        for k in range(data.shape[-1]+1)
    ])


@dataclass
class OptimumIntervalMethod:
    """Dataclass holding the necessary pre-sampled data and implementing the
    optimum interval method.

    Attributes
    ----------
    table: np.array, 3D
        lookup table of maximum intervals indexed by mu index, k, and trial index
    mu_values: np.array, 1D
        signal strength values used when generating the table above
    k_values: np.array, 1D
        k values of the table above (since k acts like an index, this is also
        np.arange(table.shape[1]))
    n_trials_per_mu: int
        number of trials that were sampled for each signal strength mu when generating
        the table, used later for normalization
    """

    table: np.array
    mu_values: np.array
    k_values: np.array
    n_trials_per_mu: int

    
    def max_signal_strength_allowed(
        self,
        data, *,
        confidence_level = 0.9
    ):
        """find the maximum signal strength that is allowed by the input data
        at the input confidence level
        
        This is an implementation of OIM, we
        - Determine the largest intervals for each possible k in the data
        - Maximize the confidence of each mu over each k using our pre-sample table of
          uniformly-distributed trials
        - Select the minimum mu whose confidence is above the input confidence level
        
        The returned array will have this result in place of the last axis of the input
        data.
    
        We are careful to respect numpy arrays and so the only assumption is that the
        input data's *last* axis is the one containing the data events. This is helpful
        since we often want to apply OIM several times over different signal hypotheses
        which lead to different CDFs and therefore different data.
    
        For example, we can broadcast the same 3 events over 200 different CDFs and then
        apply this function to find the minimum mu over those 200 different options.
    
            >>> data.shape
            (200, 3)
            >>> max_signal_strength_allowed(data).shape
            (200,)
        """
        
        # apply the largest interval algorithm and store the largest intervals in
        # an array indexed by k
        max_interval_by_k = _largest_intervals_by_k(data)
        confidence = np.full(
            (
                self._table.shape[0], # mu axis
                *max_interval_by_k.shape, # k and data axes
            ),
            0.0
        )
        for i_mu in range(self._table.shape[0]):
            for k in range(min(self._table.shape[1], max_interval_by_k.shape[0])):
                confidence[i_mu,k,...] = np.sum(
                    np.less.outer(
                        self._table[i_mu,k,:],
                        max_interval_by_k[k]
                    ),
                    axis=0
                ) / self._n_trials_per_mu
        best_conf_over_k = np.max(confidence, axis=1)
        # select mu indices where this confidence is above the input threshold
        # set the value for the mu that obtain this threshold to 1 and the others to 0
        i_mu_above_selection = np.where(best_conf_over_k > confidence_level, 1, 0)
        # ASSUMPTION: the signal strength axis is ordered by mu
        #   this allows us to find the minimum index and
        #   then use that index to find the minimum mu
        # we use argmax to find the minimum index since argmax exits at the first value that
        # obtains the maximum which we know is the first value above the confidence_level due
        # to how we constructed the array
        min_i_mu_above = np.argmax(i_mu_above_selection, axis=0)
        # now lookup the min mu using our min_i_mu
        min_mu_above = self._mu_values[min_i_mu_above]
        return min_mu_above


def load_oim_table(
    
):
    
    
    __cache_location = (
        pathlib.Path(__file__).parent.resolve()
        / 'max_interval_size_cdf_cache.pkl'
    )
    
    def __init__(
        self,
        max_signal_strength = 20.0,
        n_test_mu = 100,
        n_trials = 1_000,
        cache = True,
        force_regen = False
    ):
        if cache:
            if force_regen and OptimumIntervalStatisticCDF.__cache_location.is_file():
                OptimumIntervalStatisticCDF.__cache_location.unlink()
            if OptimumIntervalStatisticCDF.__cache_location.is_file():
                with open(OptimumIntervalStatisticCDF.__cache_location, 'rb') as cache_file:
                    self._table, self._mu_values, self._k_values, self._n_trials_per_mu = (
                        pickle.load(cache_file)
                    )
            else:
                self._table, self._mu_values, self._k_values, self._n_trials_per_mu = (
                    _generate_max_interval_samples(
                        np.linspace(0.0, max_signal_strength, n_test_mu),
                        n_trials
                    )
                )
                with open(OptimumIntervalStatisticCDF.__cache_location, 'wb') as cache_file:
                    pickle.dump(
                        (self._table, self._mu_values, self._k_values, self._n_trials_per_mu),
                        cache_file
                    )
        else:
            self._table, self._mu_values, self._k_values, self._n_trials_per_mu = (
                _generate_max_interval_samples(
                    np.linspace(0.0, max_signal_strength, n_test_mu),
                    n_trials
                )
            )
    
