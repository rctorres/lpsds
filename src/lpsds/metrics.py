from seaborn.algorithms import bootstrap

def bootstrap_estimate(vec, ci=95, n_boot=1000, seed=None):
    """
    def bootstrap_estimate(vec, ci=95, n_boot=1000)

    Returns the aggregated result for vector vec using the same CI estimator as seaborn.

    Input:
      - vec: a numpy vector [N,]
      - ci: the confidence interval to consider.
      - n_boot: how many samplings to employ when using bootstrap for the CI interval.
      - seed: the seed value to use.

    Returns: a tuple with the following values:
      - The mean value of vec
      - The lower limit of the confidence interval
      - The upper limit of the confidence interval
    """
    def percentile_interval(data, width):
        """Return a percentile interval from data of a given width."""
        edge = (100 - width) / 2
        percentiles = edge, 100 - edge
        return np.percentile(data, percentiles)

    mean = vec.mean()
    boots = bootstrap(vec, func='mean', n_boot=1000, seed=seed)
    err_min, err_max = percentile_interval(boots, ci)

    return mean, err_min, err_max
