"""Metric related tools"""

import numpy as np
from scipy.stats import gmean
from seaborn.algorithms import bootstrap
from sklearn.metrics import recall_score

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
    boots = bootstrap(vec, func='mean', n_boot=n_boot, seed=seed)
    err_min, err_max = percentile_interval(boots, ci)

    return mean, err_min, err_max


def sp_index(tp: np.array, tn: np.array) -> np.array:
  """
  def sp(tp: np.array, tn: np.array) -> np.array

  Calculates the SP index, which is given by:

  sp = \sqrt{ \sqrt{tp \times tn} \times \(\frac{tp+tn,2}\) }

  where tp is the true positive values and tn the true negative values.

  Returns: an array with the sp index calculated.
  """

  if type(tp) is np.ndarray: tp = tp.flatten()
  if type(tn) is np.ndarray: tn = tn.flatten()
  mat = np.array([tp, tn])
  return np.sqrt( gmean(mat, axis=0) * mat.mean(axis=0) ).flatten()


def sensitivity(y_true, y_pred):
  """
  def sensitivity(y_true, y_pred)

  Calculate the sensitivity score. Sklearn style.
  """
  return recall_score(y_true, y_pred, pos_label=1)


def specificity(y_true, y_pred):
  """
  def specificity(y_true, y_pred)

  Calculate the specificity score. Sklearn style.
  """
  return recall_score(y_true, y_pred, pos_label=0)


def sp_score(y_true, y_pred):
  """
  def sp_score(y_true, y_pred)

  Calculate the sp_score score. Sklearn style.
  """
  return sp_index(sensitivity(y_true, y_pred), specificity(y_true, y_pred))
