import pandas as pd


def get_operation_model(cv_model: dict, cv_splits: list, X: pd.DataFrame, y_true: pd.Series, metric:str='test_sp', less_is_better:bool=False) -> tuple:
    """"
    get_operation_model(cv_model: dict, cv_splits: list, X: pd.DataFrame, y_true: pd.Series, metric:str='test_sp', less_is_better:bool=False) -> tuple

    Returns the operation model and its corresponding data. For that, it finds
    the model yielding the best metric over its corresponding testing set.

    Input parameters:
      - cv_model: a dictionary with structure equivalent to the one returned by sklearn.model_selection.cross_validate.
      - cv_splits: a list where each item is a tuple containing the indexes to be used for training and testing in
                   see sklearn.model_selection.KFold, for isntance, for details.
      - X: a DataFrame containing **ALL** input samples used for model development and testing. It should be the same X as
           the one passed to sklearn.model_selection.cross_validate.
      - y_true: a Series containing **ALL** target samples used for model development and testing. It should be the same y_true as
           the one passed to sklearn.model_selection.cross_validate.
      - metric: a key within cv_model containing the metric to use to compare cv folds.
      - less_is_better: if True, the best fold will be the one where the passed metric is the lowest (MSE, for instance). The highest
                        will be returned if this value is set to True (Accuracy, for instance).
    
    Returns:
      - best_fold: the id of the fold yielding the operation model.
      - X: the input set used for testing in best_fold.
      - y_true: the target set used for testing in best_fold.
      - model: the operation model yielded by best_fold
    """
    best_fold = cv_model[metric].argmin() if less_is_better else cv_model[metric].argmax()
    _, tst_idx = cv_splits[best_fold]
    X = X.iloc[tst_idx]
    y_true = y_true.iloc[tst_idx]
    model = cv_model['estimator'][best_fold]
    return best_fold, X, y_true, model
