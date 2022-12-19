import collections.abc
import numpy as np
import pandas as pd
from .metrics import bootstrap_estimate

class ObjectView(dict):
    """Data structure that improves a regular map structure so
    its components can be accessed as map['value'] or map.value.
    It works only for string keys.
    """
    def __getattr__(self, name):
        if name in self: return self[name]
        else: raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self: del self[name]
        else: raise AttributeError("No such attribute: " + name)


def to_list(val):
    """
    If val is an iterable collection other than a string, returns val,
    otherwise, returs [val].
    """
    if isinstance(val, collections.abc.Iterable) and type(val) is not str:
        return val
    else: return [val]


def smart_tuple(vec):
    """
    If len(vec) == 1, returns vec[0], otherwise, returns vec as a tuple.
    """
    if len(vec) == 1: return vec[0]
    else: return tuple(vec)


def keep(df, index=None, columns=None, inplace=True):
    """
    def keep(index=None, columns=None, inplace=True)
    
    The opposite of pd.drop function. In this function, you specify the index / columns you
    want to keep. I.e. the function will drop everything that does not match the index/cools
    specified.
    
    IMPORTANT REMARKS:
     - differently than the drop function, "inplace" optioon defaults to True.
     - index and columns are FILTER, not the final list of values to be kept.
    
    Returns: the filtered dataframe if inplace=True or None otherwise.
    """
    
    if index is not None:
        index = df.loc[~index].index
    
    if columns is not None:
        columns = df.columns[~columns]
    
    return df.drop(index=index, columns=columns, inplace=inplace)


def confusion_matrix_annotation(cm: np.array, fmt_str: str='${mean:.2f}^{{+{e_max:.2f}}}_{{-{e_min:.2f}}}$', use_difference: bool=True, seed: int=None) -> pd.DataFrame:
    """
    def confusion_matrix_annotation(cm: np.array, fmt_str: str='${mean:.2f}^{{+{e_max:.2f}}}_{{-{e_min:.2f}}}$', 
                                    use_difference: bool=True, seed: int=None) -> pd.DataFrame
    
    Creates a pandas dataframe with fancy annotation to be used in seaborn.heatmap. It automatically calculates
    mean and error margins using lpsds.metrics.bootstrap_estimate.

    Input parameters:
      - cm: a numpy.array object with the confusion matrix organized as [num_folds, num_rows, num_cols].
      - fmt_str: a tring defining how the numbers should be presented. Defaults to mean^{+err max}_{-err_min}.
                 you must follow the right name convention if you want to use your own fmt string. Use the convention as:
                    - mean: the mean value
                    - e_min: the lower threshold error margin
                    - e_max: the higher threshold error margin.
      - use_difference: if True (default), will present e_min as (mean-e_min) and e_max as (e_max - mean).
      - seed: random seed.

    Returns: a dataframe with shape [num_rows, num_cols] containing in each cell the defined annotation.

    ATTENTION: when passing the annotation dataframe to seaborn.heatmap, you MUST set fmt=''. Example:
               seaborn.heatmap(annot=my_annotation_df, fmt='') 
    """
    
    num_rows, num_cols = cm.shape[1:]
    ret = pd.DataFrame(np.empty([num_rows, num_cols], dtype=str))
    for r in range(num_rows):
        for c in range(num_cols):
            mean, e_min, e_max = bootstrap_estimate(cm[:,r,c], seed=seed)
            if use_difference:
                e_min = mean - e_min
                e_max -= mean
            ret.iloc[r][c] = fmt_str.format(mean=mean, e_min=e_min, e_max=e_max)
    
    return ret