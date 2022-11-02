import os
import re
import pandas as pd
import tempfile
import mlflow
import lpsds.metrics

def log_dataframe(df: pd.DataFrame, var_name: str, folder: str=''):
    """"
    def log_dataframe(df, var_name, folder='')

    Saves a pandas dataframe to MLFlow as a .parquet file.

    Input parameters:
      - df: the pandas.DataFrame object to be saved.
      - var_name: the name the dataframe will have within MLFlow.
      - folder: the path (in MLFlow) to where the dataframe will be saved.
    """

    with tempfile.TemporaryDirectory() as temp_path:
        temp_file_name = os.path.join(temp_path, var_name + '.parquet')
        df.to_parquet(temp_file_name, index=False)
        mlflow.log_artifact(temp_file_name, folder)


def log_statistics(cv_model: dict) -> dict:
    """
    def log_statistics(cv_model)

    Log metrics obtained via cross validation to MLFlow as statistical summaries.
    I.e.: the mean value obtained across all folds and its minimum and maximum
    values so right value will be found in this range (err_min, mean, err_max)
    with 95% C.I.

    The metrics will be found automatically by looking to keys within cv_model
    object that has the pattern "test_*".

    Input:
        cv_model: the result of the sklearn cross_validate function.
    
    Returns a map where keys are the metric name and values are its mean, err min and err max
    """
    regexp = re.compile(r'test_(.+)')
    ret_map = {}
    for metric, values in cv_model.items():
        grp = regexp.match(metric)
        if grp is not None:
            mean, err_min, err_max = lpsds.metrics.bootstrap_estimate(values)
            metric_name = grp.group(1)
            ret_map[metric_name] = {
                f'{metric_name}_err_min' : err_min,
                f'{metric_name}_mean' : mean,
                f'{metric_name}_err_max' : err_max
            }
            mlflow.log_metrics(ret_map[metric_name])
    
    return ret_map
