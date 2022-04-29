"""
Tools for model handling
"""

from sklearn.metrics import mean_squared_error
import numpy as np


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