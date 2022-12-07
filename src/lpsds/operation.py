import numpy as np
import pandas as pd
from typing import Union


def get_fold_data(cv_model: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], fold_idx: int) -> tuple:
     """"
     def get_fold_data(cv_model: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], fold_idx: int) -> tuple

     Returns the data corresponding to a given cross validation fold.

     Input parameters:
       - cv_model: a dictionary with structure equivalent to the one returned by sklearn.model_selection.cross_validate.
       - cv_splits: a list where each item is a tuple containing the indexes to be used for training and testing in
                    see sklearn.model_selection.KFold, for isntance, for details.
       - X: a DataFrame containing **ALL** input samples used for model development and testing. It should be the same X as
            the one passed to sklearn.model_selection.cross_validate.
       - y_true: a Series containing **ALL** target samples used for model development and testing. It should be the same y_true as
            the one passed to sklearn.model_selection.cross_validate.
       - fold_idx: the index of the cv fold you want to collect the data.
    
     Returns:
       - X_train: the input set used for training in the desired fold.
       - X_test: the input set used for testing in the desired fold.
       - y_train: the target set used for training in the desired fold.
       - y_test: the target set used for testing in the desired fold.
       - model: the operation model yielded by the desired fold.
     """
     trn_idx, tst_idx = cv_splits[fold_idx]

     if hasattr(X, 'iloc'):
          X_train = X.iloc[trn_idx]
          X_test = X.iloc[tst_idx]
     else:
          X_train = X[trn_idx,]
          X_test = X[tst_idx,]

     if hasattr(y_true, 'iloc'):
          y_train = y_true.iloc[trn_idx]
          y_test = y_true.iloc[tst_idx]
     else:
          y_train = y_true[trn_idx,]
          y_test = y_true[tst_idx,]
    
     model = cv_model['estimator'][fold_idx]
    
     return X_train, X_test, y_train, y_test, model



def get_operation_model(cv_model: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], metric:str='test_sp', less_is_better:bool=False) -> tuple:
    """"
    get_operation_model(cv_model: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], metric:str='test_sp', less_is_better:bool=False) -> tuple

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
    _, X_tst, _, y_tst, model = get_fold_data(cv_model, cv_splits, X, y_true, best_fold)
    return best_fold, X_tst, y_tst, model


def get_staged_metrics(cv: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], metrics_map: dict, **kwargs) -> pd.DataFrame:
     """
     def get_staged_metrics(cv: dict, cv_splits: list, X: Union[pd.DataFrame, np.array], y_true: Union[pd.Series, np.array], metrics_map: dict, **kwargs) -> pd.DataFrame:

     Evaluates a set of metrics for each stage in an enseble model taking into consideration the cross-validation folds employed for the model development.

     Input parameters:
      - cv: a dictionary with structure equivalent to the one returned by sklearn.model_selection.cross_validate.
      - cv_splits: a list where each item is a tuple containing the indexes to be used for training and testing in
                   see sklearn.model_selection.KFold, for isntance, for details.
      - X: a DataFrame containing **ALL** input samples used for model development and testing. It should be the same X as
           the one passed to sklearn.model_selection.cross_validate.
      - y_true: a Series containing **ALL** target samples used for model development and testing. It should be the same y_true as
                the one passed to sklearn.model_selection.cross_validate.
      - metrics_map: a map where key is a string label identifying the metric to be evaluated at each stage, and value is a reference
                     to a function in the format metric_function(y_pred, y_true, **kwargs) -> float that will be responsible to evaluate
                     the desired metric for each stage of the ensemble model for each cv-fold.

     Returns: a pandas.DataFrame with the following columns:
       - Metric: the metric name (given by the keys in metrics_map)
       - Fold: the cv-fold ID where the metric is being evaluated.
       - Stage: the number of stages considered when evaluating the metric.
       - Value: the achieved metric value.
     """

     aux_df_list = []
     num_folds = len(cv_splits)
     for fold in range(num_folds):
         #Getting the test data and model for the fold
         _, in_tst, _, targ_tst, model = get_fold_data(cv, cv_splits, X, y_true, fold)

         #Pipelines do not implemet staged_predict. As a result, we need
         #to apply the pre-processing manually. Probably there is a better way to do this...
         for _, pp in model.steps[:-1]:
             if hasattr(pp, 'transform'):
                 in_tst = pp.transform(in_tst)

         #Collecting just the final estimator (i.e. the model being applied right after all the pre-processing pipeline).
         estimator = model.steps[-1][1]

         #Calculating the accuracy for each class w.r.t the number of stages.
         for stage, out_tst in enumerate(estimator.staged_predict(in_tst)):
             for metric_name, metric_function in metrics_map.items():
                 metric_value = metric_function(targ_tst, out_tst, **kwargs)
                 #Saving the results in an auxiliary dataframe
                 aux_df = pd.DataFrame({
                     'Metric' : [metric_name],
                     'Value' : [metric_value],
                     'Stage' : [stage],
                     'Fold' : [fold],
                 })
                 aux_df_list.append(aux_df)

     #Creating the final dataframe in a format suitable for seaborn (long-format)
     return pd.concat(aux_df_list)
 