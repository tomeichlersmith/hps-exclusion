"""numpy implementation of the optimum interval method

Yellin (2003) https://arxiv.org/abs/physics/0203002
"""


import numpy as np


def _load_lookup_table():
    """load the max intervals (separated by signal strength mu and number of events k)
    lookup table from the pickle file

    We look for the file in the same directory as this file and return
    the lookup table for later usage.
    """
    import pathlib
    import pickle
    with open(pathlib.Path(__file__).parent.resolve() / 'interval_ntrials_10000.p', 'rb') as f:
         return pickle.load(f)


def k_largest_intervals(cumulants):
    """generate list of largest intervals according to the input cumulants

    The index of this list is k or the number of events within those intervals.

    We assume that cumulants are already sorted and transformed into cumulants using a CDF
    (as the name implies).
    """
    # assume the last axis is the one for the cumulants of the events
    cumulants_with_edges = np.full((*cumulants.shape[:-1], cumulants.shape[-1]+2), 0.)
    cumulants_with_edges[...,1:-1] = cumulants
    cumulants_with_edges[...,-1] = 1.
    return [
        np.max(cumulants_with_edges[...,(k+1):]-cumulants_with_edges[...,:-1*(k+1)], axis=-1)
        for k in range(cumulants.shape[-1]+1)
    ]


def max_signal_strength_allowed(
    data_cumulants, *,
    confidence_level = 0.9
):
    """find the maximum signal strength that is allowed by the input data
    at the input confidence level
    
    This is an implementation of OIM, we
    - Determine the largest intervals for each possible k in the cumulants
    - Lookup the confidence of all the possible mu, k pairs in the lookup table
    - Maximize the confidence of each mu over each k
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
        >>> min_mu_with_best_k_over(data_cumulants).shape
        (200,)
    """

    if getattr(max_signal_strength_allowed, 'max_interval_trails_lut', None) is None:
        max_signal_strength_allowed.max_interval_trails_lut = _load_lookup_table()
        max_signal_strength_allowed.tested_signal_strengths = np.array(
            list(max_signal_strength_allowed.max_interval_trails_lut.keys())
        )
    
    # apply the largest interval algorithm and store the largest intervals in
    # an array indexed by k
    max_interval_by_k = k_largest_intervals(data_cumulants)

    # confidence is a 3D array whose indices are
    # 0th -> index of mu in lookupTable
    # 1st -> k
    # the rest -> other (non event) indices of data_cumulants array
    confidence = np.full(
        (
            len(max_signal_strength_allowed.max_interval_trails_lut),
            len(max_interval_by_k),
            *data_cumulants.shape[:-1]
        ),
        0.
    )
    for i_mu, mu in enumerate(max_signal_strength_allowed.max_interval_trails_lut):
        for k in range(len(max_interval_by_k)):
            confidence[i_mu,k,:] = np.sum(
                np.less.outer(
                    max_signal_strength_allowed.max_interval_trails_lut[mu][k],
                    max_interval_by_k[k]
                ),
                axis=0
            )/10000 if k in max_signal_strength_allowed.max_interval_trails_lut[mu] else 0.
    # find k with best confidence for each mu
    best_conf_over_k = np.max(confidence, axis=1)
    # select mu indices where this confidence is above the input threshold
    # set the value for the mu that obtain this threshold to 1 and the others to 0
    i_mu_above_selection = np.where(best_conf_over_k > confidence_level, 1, 0)
    # ASSUMPTION: the lookupTable is ordered by mu
    #   this allows us to find the minimum index and then use that index to find the minimum mu
    # we use argmax to find the minimum index since argmax exits at the first value that
    # obtains the maximum which we know is the first value above the confidence_level due
    # to how we constructed the array
    min_i_mu_above = np.argmax(i_mu_above_selection, axis=0)
    # now lookup the min mu using our min_i_mu
    min_mu_above = max_signal_strength_allowed.tested_signal_strengths[min_i_mu_above]
    return min_mu_above