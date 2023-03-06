import collections.abc
import numpy as np
import pandas as pd
import sklearn.pipeline
from typing import Union
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



def pretty_title(text :str, abbrev_threshold :int=4) -> str:
    """
    def pretty_title(text :str, abbrev_threshold :int=4) -> str:

        Returns a pretty string. If it is an abbreviation (text > than abbrev_threshold),
        the ext will be put in upper case. Otherwise, it will have all its first letters
        in upper case the rest in lower case.
    """
    return text.upper() if len(text) <= abbrev_threshold else text.capitalize()



def error_bar_roc(df :pd.DataFrame, tp_col_name :str='true_positive', fp_col_name: str='false_positive', 
                    fold_col_name :str='fold', threshold_col_name :str='threshold', num_points: int=100) -> pd.DataFrame:
    """
    def error_bar_roc(df :pd.DataFrame, tp_col_name :str='true_positive', fp_col_name: str='false_positive', 
                        fold_col_name :str='fold', threshold_col_name :str='threshold', num_points: int=100) -> pd.DataFrame:

    Creates a new ROC curve for each CV fold, making sure that all folds share the same false alarm values, allowing plotting
    ROC curve with its error marging.

    Input:
      - df: a dataframe containing the ROC values in long format (fold x tpr x fpr x threshold). All mentioned ROC values
            must be present in df.
      - tp_col_name: the name of the column in df holding the ROC true positive values.
      - fp_col_name: the name of the column in df holding the ROC false positive values.
      - fold_col_name: the name of the column in df holding the fold ID of each ROC value.
      - threshold_col_name: the name of the column in df holding the ROC threshold values.
      - num_points: how many points will be used to compute the new ROCS.
    
    Return: a dataframe where all TP values (for all folds) were computed for the same FP values,
            so all ROCS have the save FP values and therefore, the same length (num_points).
            Columns in the new dataframe are::
                - tp_col_name: the name of the column in df holding the new ROC true positive values.
                - fp_col_name: the name of the column in df holding the new ROC false positive values.
                - fold_col_name: the name of the column in df holding the fold ID of each new ROC value.
    """
    #false alarm must be in ascending order otherwise np.interp will not work.
    df = df.sort_values([fold_col_name, threshold_col_name], ascending=False)
    
    num_folds = len(df[fold_col_name].unique())
    mean_fpr = np.linspace(0, 1, num_points)
    
    rocs_list = []
    for fold in range(num_folds):
        fold_roc = df.loc[df[fold_col_name] == fold]
        interp_tpr = np.interp(mean_fpr, fold_roc[fp_col_name], fold_roc[tp_col_name])
        roc_aux = pd.DataFrame({fp_col_name : mean_fpr, tp_col_name : interp_tpr, fold_col_name : fold})
        rocs_list.append(roc_aux)
    
    return pd.concat(rocs_list, ignore_index=True)



def pipeline_split(model :sklearn.pipeline.Pipeline, X: Union[pd.DataFrame, np.ndarray]=None):
    """
    def pipeline_input_split(model :sklearn.pipeline.Pipeline, X: Union[pd.DataFrame, np.ndarray]=None):
    
    Breaks a pipeline into 2 sections: preprocessing and estimator sections. If X is provided, the function
    will apply the pre-processing chain on it, and return it in a stage right before it would be fed to the
    pipeline estimator.

    Input:
      - model: a sklearn-li Pipeline object
      - X: an input dataset.
    
      Returns:
        - A list with the pipeline pre-processing steps.
        - The estimator object at the end of the pipeline.
        - Pre-processed X (the values of X right before it will be fed to the pipeline estimator)
    """
    
    pre_processing = model.steps[:-1]
    estimator = model.steps[-1]
    
    if X is not None:
        for _, func in pre_processing:
            X = func.transform(X)
    
    return pre_processing, estimator, X
