"""
Tools for model handling
"""

from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from typing import Union


def get_operation_model(model, X, y, metric=mean_squared_error, **kargv):
    """
    def get_operation_model(model, X, y, metric=mean_squared_error, *argv)

    Selects an operation model by running the entire dataset for each fold
    and picking the one with the best metric.

    The function always assume that the best value for a metric is its minimum value
    (MSE, for instance). If you use a metric like R2, where the higher the better, you must
    multiply such metric by -1.

    Input parameters:
        - model: a model structure as returned by sklearn cross_validation.
        - X: the input dataset
        - y: the target values
        - metric: a metric to be used (MSE, R2, etc.)
        - **kargv: passed down to the metric function.
    """
    res = np.array([metric(y, mod.predict(X), **kargv) for mod in model['estimator']])
    op_idx = np.argmin(res)
    return model['estimator'][op_idx]


def create_validation_dataset(model, X, y, proba: bool=True, copy: bool=True):
    """
    def create_validation_dataset(model, X, y, proba: bool=True, copy: bool=True)

    Create a validation dataset suitable for validate your model
    when deployed to another operation environment.

    Input parameters are:
        - model: a sklearn-like model (pipeline, skorch, etc.)
        - X: the input dataset
        - y: the targets of X
        - proba: if True, will call model.predict_proba. Otherwise, model.predict
        - copy: whether to return a copy of the passed dataset (both X and y).
    
    Returns:
        A new dataset containing 2 new columns:
            - y_true: containing the targets in y.
            - y_pred: the model output for each sample in X.
    """

    if copy: X = X.copy()
    X['y_true'] = y
    X['y_pred'] = model.predict_proba(X) if proba else model.predict(X)
    return X


def get_input_variables_description(X):
    """
    Create a descriptive pandas telling the following info from X:
      - The order of each column (Inpout Order)
      - Column names (Input Name)
      - Column types (Input Type)
    """
    input_description = X.dtypes.to_frame().reset_index()
    input_description.index.name = 'Input Order'
    input_description.columns = ['Input Name', 'Input Type']
    return input_description


def feature_importances(model, X :Union[pd.DataFrame, np.ndarray], y_true :Union[pd.Series, np.ndarray], suppressing_function=np.mean,
                        metric_function=mean_squared_error, comparison_function=np.subtract) -> pd.DataFrame:
    """
        def feature_importances(model, X :Union[pd.DataFrame, np.ndarray], y_true :Union[pd.Series, np.ndarray], suppressing_function=np.mean,
                                metric_function=mean_squared_error, comparison_function=np.subtract) -> pd.DataFrame:
    
    Calculate feature importance using variables suppressing method. The function will suppress
    ove variable at a time and calculate the difference in a provided metric w.r.t using all features.
    The higher the difference, the higher the variable importance.

    Input:
      - model: a trained model defining the predict_proba method.
      - X: model input set.
      - y_true: true labes of X.
      - suppressing_function: the function to be used to suppress the feature. It must receive the feature and
                              must return a new unique value to represent the feature. Default to arithmetic mean.
      - metric_function: the metric to be used to evaluate model's performance. Must receive y_true and y_pred and return a scalar.
      - comparison_function: a function that will compare the metric obtained when suppressing a metric and
                             the baseline (metric value when no feature is suppressed). Must receive 2 scalars and return a scalar.
    
    Returns: a pandas DataFrame where the index is the feature name and the column ('improtance') is the metric deviation
              after suppressing the given feature.
    """

    baseline = metric_function(y_true, model.predict(X))

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=np.arange(X.shape[1]))

    ret = {}
    for c in X.columns:
        aux = X.copy()
        aux[c] = suppressing_function(aux[c])
        relev_metric = metric_function(y_true, model.predict(aux))
        ret[c] = comparison_function(baseline, relev_metric)
    
    ret_df = pd.DataFrame.from_dict(ret, orient='index', columns=['importance'])
    ret_df.index.set_names('feature', inplace=True)
    return ret_df
