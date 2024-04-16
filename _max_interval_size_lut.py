"""module for generating, saving, and loading the max interval size LUT"""

import numpy as np
import pathlib
import pickle


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


class MaxIntervalSizeCDF:
    """In-memory generation and lookup of the cumulative distribution function (CDF)
    of the maximum interval size partitioned by signal strength ($\mu$) and number of
    events within the interval ($k$).

    To Do
    -----
    - improve caching behavior?
    """

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
            if force_regen and MaxIntervalSizeCDF.__cache_location.is_file():
                MaxIntervalSizeCDF.__cache_location.unlink()
            if MaxIntervalSizeCDF.__cache_location.is_file():
                with open(MaxIntervalSizeCDF.__cache_location, 'rb') as cache_file:
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
                with open(MaxIntervalSizeCDF.__cache_location, 'wb') as cache_file:
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
    
    @staticmethod
    def _k_largest_intervals(cumulants):
        """generate list of largest intervals according to the input cumulants
    
        The index of this list is k or the number of events within those intervals.
    
        We assume that cumulants are already sorted and transformed into cumulants using a CDF
        (as the name implies).
        """
        # assume the last axis is the one for the cumulants of the events
        cumulants_with_edges = np.full((*cumulants.shape[:-1], cumulants.shape[-1]+2), 0.)
        cumulants_with_edges[...,1:-1] = cumulants
        cumulants_with_edges[...,-1] = 1.
        return np.array([
            np.max(cumulants_with_edges[...,(k+1):]-cumulants_with_edges[...,:-1*(k+1)], axis=-1)
            for k in range(cumulants.shape[-1]+1)
        ])
    
        
    def max_signal_strength_allowed(
        self,
        data_cumulants, *,
        confidence_level = 0.9
    ):
        """find the maximum signal strength that is allowed by the input data
        at the input confidence level
        
        This is an implementation of OIM, we
        - Determine the largest intervals for each possible k in the cumulants
        - Maximize the confidence of each mu over each k using our pre-calculated CDF LUT
        - Select the minimum mu whose confidence is above the input confidence level
        The returned array will have this result in place of the last axis of the input
        data_cumulants.
    
        We are careful to respect numpy arrays and so the only assumption is that the
        input data's *last* axis is the one containing the data events. This is helpful
        since we often want to apply OIM several times over different signal hypotheses
        which lead to different CDFs and therefore different data cumulants.
    
        For example, we can broadcast the same 3 events over 200 different CDFs and then
        apply this function to find the minimum mu over those 200 different options.
    
            >>> data_cumulants.shape
            (200, 3)
            >>> max_signal_strength_allowed(data_cumulants).shape
            (200,)
        """
        
        # apply the largest interval algorithm and store the largest intervals in
        # an array indexed by k
        max_interval_by_k = MaxIntervalSizeCDF._k_largest_intervals(data_cumulants)
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