import os
import re
import numpy as np
import pandas as pd
import tempfile
import mlflow
import lpsds.metrics
from lpsds.utils import ObjectView


class MLFlow:
    def __init__(self, run_id=None):
        active_run = mlflow.active_run()

        if run_id is None and active_run is None:
            raise ValueError('You must invoke this class either by passing a valid run id or within an started run (mlflow.start_run)')

        self.run_id = active_run.info.run_id if run_id is None else run_id
        self.run = mlflow.get_run(self.run_id)
        self.run_client = mlflow.tracking.MlflowClient()

    def log_dataframe(self, df: pd.DataFrame, var_name: str, folder: str=''):
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


    def log_statistics(self, cv_model: dict) -> dict:
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
                for i,v in enumerate(values): mlflow.log_metric(metric_name, v, step=i)
        
        return ret_map


    def get_params(self):
        """
        Must return model parameters, such as:
          - database
          - best_fold_id
        """
        return self.run.data.params

    def get_run_id(self) -> str:
        """
        def get_run_id(self)

        Returns the experiment´s run id.
        """
        return self.run_id


    def get_metrics(self, as_dict=False):
        """
        Collects all metrics available for a given run.
        

        Returns a pandas dataframe with all metrics.

        if as_dict is True, the method will return the metrics as
        a dict, where, If the metric is a vector, it is returned as a numpy.array.
        """
        
        #Collecting all metrics in a dataframe
        df = pd.DataFrame(columns=['metric', 'step', 'value'])
        for metric_name in self.run.data.metrics.keys():
            for m in self.run_client.get_metric_history(self.run_id, metric_name):
                df.loc[len(df)] = metric_name, m.step, m.value
        df.sort_values(['metric', 'step'], inplace=True, ignore_index=True)
        
        if not as_dict: return df
    
        ret = ObjectView()
        #Creating a dataframe where vectorized metrics are saved as lists
        grp = df.groupby('metric').value.agg(lambda x: x.iloc[0] if len(x) == 1 else x.to_list()).to_dict()
        
        #Lists are converted to np.arrays
        for k,v in grp.items():
            if hasattr(v, '__iter__'):
                grp[k] = np.array(v)
        return grp



    def get_dataframe(self, var_name: str, folder: str='') -> pd.DataFrame:
        """"
        def get_dataframe(self, var_name: str, folder: str='') -> pd.DataFrame:

        Load a pandas dataframe saved to MLFlow as a .parquet file.

        Input parameters:
        - var_name: the name the dataframe have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be loaded from.
        
        Return a pandas.DataFrame with the collected info.
        """

        with tempfile.TemporaryDirectory() as temp_path:
            full_path = os.path.join(folder, var_name + '.parquet')
            local_path = self.run_client.download_artifacts(self.run_id, full_path, temp_path)
            return pd.read_parquet(local_path)
