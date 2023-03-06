import os
import re
import yaml
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

    def log_dataframe(self, df: pd.DataFrame, var_name: str, folder: str='') -> None:
        """"
        def log_dataframe(self, df: pd.DataFrame, var_name: str, folder: str='') -> None

        Saves a pandas dataframe to MLFlow as a .parquet file.

        Input parameters:
        - df: the pandas.DataFrame object to be saved.
        - var_name: the name the dataframe will have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be saved.
        """

        self.log_artifact(df, var_name + '.parquet', folder, save_func=df.to_parquet, object_param_name=None, fname_param_name='path')


    def log_numpy(self, mat: np.ndarray, var_name: str, folder: str='') -> None:
        """"
        def log_numpy(self, mat: np.ndarray, var_name: str, folder: str='') -> None

        Saves a numpy matrix to MLFlow as a .np file.

        Input parameters:
        - mat: the np.ndarray object to be saved.
        - var_name: the name the dataframe will have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be saved.
        """

        self.log_artifact(mat, var_name + '.npy', folder, allow_pickle=False)



    def log_artifact(self, obj, var_name: str, folder: str='', save_func=np.save,
                     object_param_name='arr', fname_param_name='file', **save_func_kwargs) -> None:
        """"
        def log_artifact(self, obj, var_name: str, folder: str='', save_func=np.save,
                         object_param_name='arr', fname_param_name='file', **save_func_kwargs) -> None:

        Saves an object of any type to mlflow.

        Input parameters:
        - obj: the object you want to save to MLFlow.
        - var_name: the name the dataframe will have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be saved.
        - save_func: a reference to a function which is responsible to save the passed object to disk (obj.to_parquet, np.save, etc.)
        - object_param_name: the save_func parameter name used to receive obj when saving (arr in np.save, for instance).
        - fname_param_name: the save_func parameter name used to receive the path where the file will be saved (path in np.save, for instance).
        - save_func_kwargs: additional parameters to be passed to save_func.
        """

        with tempfile.TemporaryDirectory() as temp_path:
            temp_file_name = os.path.join(temp_path, var_name)
            save_func_kwargs[fname_param_name] = temp_file_name
            if object_param_name is not None:
                save_func_kwargs[object_param_name] = obj
            save_func(**save_func_kwargs)
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


    def get_params(self, infer_types: bool=False) -> ObjectView:
        """
        Returns model parameters.

        Input:
          - infer_dtypes: if True, will try to infer values types, since mlflow 
                          store them as strings.
        
        Return: a map with the model parameters.
        """
        ret = ObjectView(self.run.data.params)
        if infer_types:
            for k,v in ret.items():
                ret[k] = yaml.safe_load(v)
        return ret


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
    
        #Creating a dataframe where vectorized metrics are saved as lists
        grp = df.groupby('metric').value.agg(lambda x: x.iloc[0] if len(x) == 1 else x.to_list()).to_dict()
        
        #Lists are converted to np.arrays
        for k,v in grp.items():
            if hasattr(v, '__iter__'):
                grp[k] = np.array(v)
        return ObjectView(grp)



    def get_dataframe(self, var_name: str, folder: str='') -> pd.DataFrame:
        """"
        def get_dataframe(self, var_name: str, folder: str='') -> pd.DataFrame:

        Load a pandas dataframe saved to MLFlow as a .parquet file.

        Input parameters:
        - var_name: the name the dataframe have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be loaded from.
        
        Return a pandas.DataFrame with the collected info.
        """

        return self.get_artifact(var_name + '.parquet', folder, pd.read_parquet)


    def get_numpy(self, var_name: str, folder: str='') -> np.ndarray:
        """"
        def get_numpy(self, var_name: str, folder: str='') -> np.ndarray:

        Load a numpy.ndarray object saved to MLFlow as a .parquet file.

        Input parameters:
        - var_name: the name the dataframe have within MLFlow.
        - folder: the path (in MLFlow) to where the dataframe will be loaded from.
        
        Return a np.ndarray with the collected info.
        """

        return self.get_artifact(var_name + '.npy', folder, np.load, allow_pickle=False)



    def get_artifact(self, var_name: str, folder: str='', load_func=np.load, **load_func_kwargs):
        """"
        def get_artifact(self, var_name: str, folder: str='', load_func=np.load, **load_func_kwargs)

        Load an artifact using the provided loading function.

        Input parameters:
        - var_name: the name the dataframe have within MLFlow (must include extensions, if existing).
        - folder: the path (in MLFlow) to where the dataframe will be loaded from.
        - load_func: the function used to load the required artifact (pd.read_parquet, np.load, etc).
        - load_func_kwargs: additional aprameters to be passed to load_func
        
        Returns whatever load_func returns.
        """

        with tempfile.TemporaryDirectory() as temp_path:
            full_path = os.path.join(folder, var_name)
            local_path = mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=full_path, dst_path=temp_path)
            return load_func(local_path, **load_func_kwargs)




    def get_experiment(self) -> mlflow.entities.Experiment:
        """
        def get_experiment(self) -> mlflow.entities.Experiment

        Returns an instance to the experiment that contains the class given mlflow run id.
        """
        exp_id = self.run.info.experiment_id
        return mlflow.get_experiment(exp_id)