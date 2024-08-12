"""fitting and averaging"""

import numpy as np
import scipy
import hist


# first I write my own mean calculation that includes the possibility of weights
#   and returns the mean, standard deviation, and the error of the mean
def weightedmean(values, weights = None) :
    """calculate the weighted mean and standard deviation of the input values
    
    This function isn't /super/ necessary, but it is helpful for the itermean
    function below where the same code needs to be called in multiple times.
    """ 
    mean = np.average(values, weights=weights)
    stdd = np.sqrt(np.average((values-mean)**2, weights=weights))
    merr = stdd/np.sqrt(weights.sum())
    return mean, stdd, merr

# now I can write the iterative mean
def itermean(values, weights = None, *, sigma_cut = 3.0) :
    """calculate an iterative mean and standard deviation

    If no weights are provided, then we assume they are all one.
    The sigma_cut parameter is what defines what an outlier is.
    If a sample is further from the mean than the sigma_cut times
    the standard deviation, it is removed.
    """
    mean, stdd, merr = weightedmean(values, weights)
    num_included = len(values)+1 # just to get loop started
    # first selection is all non-zero weighted samples
    selection = (weights > 0) if weights is not None else np.full(len(values), True)
    while np.count_nonzero(selection) < num_included :
        # update number included for this mean
        num_included = np.count_nonzero(selection)
        # calculate mean and std dev
        mean, stdd, merr = weightedmean(values[selection], weights[selection] if weights is not None else None)
        # determine new selection, since this variable was defined outside
        #   the loop, we can use it in the `while` line and it will just be updated
        selection = (values > (mean - sigma_cut*stdd)) & (values < (mean + sigma_cut*stdd))

    # left loop, meaning we settled into a state where nothing is outside sigma_cut standard deviations
    #   from our mean
    return mean, stdd, merr


def scaled_normal(x, mean, stdd, scale):
    return scale*scipy.stats.norm.pdf(x, mean, stdd)


def fit_histogram(histogram: hist.Hist, f, **kwargs):
    x    = histogram.axes[0].centers
    y    = histogram.values()
    yerr = np.sqrt(
        histogram.variances()
        if histogram.variances() is not None else
        histogram.values() # assume Poisson errors if no variances
    )

    x = x[yerr > 0]
    y = y[yerr > 0]
    yerr = yerr[yerr > 0]

    return scipy.optimize.curve_fit(
        f,
        x,
        y,
        sigma = yerr,
        absolute_sigma = True,
        **kwargs
    )


def fitnorm(histogram: hist.Hist):
    return fit_histogram(histogram, scaled_normal)[0]