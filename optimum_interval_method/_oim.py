"""module for generating, saving, and loading the max interval size LUT"""

from dataclasses import dataclass


import numpy as np


def largest_intervals_by_k(data):
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
        max_interval_by_k = largest_intervals_by_k(data)
        confidence = np.full(
            (
                self.table.shape[0], # mu axis
                *max_interval_by_k.shape, # k and data axes
            ),
            0.0
        )
        for i_mu in range(self.table.shape[0]):
            for k in range(min(self.table.shape[1], max_interval_by_k.shape[0])):
                confidence[i_mu,k,...] = np.sum(
                    np.less.outer(
                        self.table[i_mu,k,:],
                        max_interval_by_k[k]
                    ),
                    axis=0
                ) / self.n_trials_per_mu
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
        min_mu_above = self.mu_values[min_i_mu_above]
        return min_mu_above
