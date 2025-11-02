import os
import re
import sklearn
import yaml
import numpy as np
import pandas as pd
import tempfile
import mlflow
import lpsds.metrics
from typing import Dict, Optional, Any, Callable, Union
from lpsds.utils import ObjectView
from sklearn.utils import estimator_html_repr
from sklearn.pipeline import Pipeline


class MLFlow:
    """MLFlow utilities class for logging and retrieving artifacts, metrics, and models."""
    
    def __init__(self, run_id: Optional[str] = None) -> None:
        """Initialize MLFlow instance.
        
        Args:
            run_id: MLFlow run ID. If None, uses the currently active run.
            
        Raises:
            ValueError: If no run_id provided and no active run exists.
        """
        active_run = mlflow.active_run()

        if run_id is None and active_run is None:
            raise ValueError('You must invoke this class either by passing a valid run id or within an started run (mlflow.start_run)')

        self.run_id = active_run.info.run_id if run_id is None else run_id
        self.run = mlflow.get_run(self.run_id)
        self.run_client = mlflow.tracking.MlflowClient()

    def log_dataframe(self, df: pd.DataFrame, var_name: str, folder: str = '') -> None:
        """Save a pandas DataFrame to MLFlow as a parquet file.

        Args:
            df: The pandas DataFrame object to be saved.
            var_name: The name the dataframe will have within MLFlow.
            folder: The path (in MLFlow) to where the dataframe will be saved.
        """
        self.log_artifact(df, var_name + '.parquet', folder, save_func=df.to_parquet, object_param_name=None, fname_param_name='path')

    def log_numpy(self, mat: np.ndarray, var_name: str, folder: str = '') -> None:
        """Save a numpy array to MLFlow as a .npy file.

        Args:
            mat: The numpy array object to be saved.
            var_name: The name the array will have within MLFlow.
            folder: The path (in MLFlow) to where the array will be saved.
        """
        self.log_artifact(mat, var_name + '.npy', folder, allow_pickle=False)

    def log_artifact(self, obj: Any, var_name: str, folder: str = '', save_func: Callable = np.save,
                     object_param_name: Optional[str] = 'arr', fname_param_name: str = 'file', 
                     **save_func_kwargs: Any) -> None:
        """Save an object of any type to MLFlow.

        Args:
            obj: The object you want to save to MLFlow.
            var_name: The name the object will have within MLFlow.
            folder: The path (in MLFlow) to where the object will be saved.
            save_func: A function responsible for saving the object to disk.
            object_param_name: The save_func parameter name for the object. If None, 
                             obj is not passed as a named parameter.
            fname_param_name: The save_func parameter name for the file path.
            **save_func_kwargs: Additional parameters to be passed to save_func.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            temp_file_name = os.path.join(temp_path, var_name)
            save_func_kwargs[fname_param_name] = temp_file_name
            if object_param_name is not None:
                save_func_kwargs[object_param_name] = obj
            save_func(**save_func_kwargs)
            mlflow.log_artifact(temp_file_name, folder)

    def log_statistics(self, cv_model: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Log cross-validation metrics to MLFlow as statistical summaries.
        
        Automatically finds metrics with the pattern "test_*" and logs their mean,
        minimum, and maximum values with 95% confidence intervals.

        Args:
            cv_model: The result of the sklearn cross_validate function.
        
        Returns:
            Dictionary mapping metric names to their statistical summaries 
            (mean, err_min, err_max).
        """
        regexp = re.compile(r'test_(.+)')
        ret_map = {}
        for metric, values in cv_model.items():
            grp = regexp.match(metric)
            if grp is not None:
                mean, err_min, err_max = lpsds.metrics.bootstrap_estimate(values)
                metric_name = grp.group(1)
                ret_map[metric_name] = {
                    f'{metric_name}_err_min': err_min,
                    f'{metric_name}_mean': mean,
                    f'{metric_name}_err_max': err_max
                }
                mlflow.log_metrics(ret_map[metric_name])
                for i, v in enumerate(values): 
                    mlflow.log_metric(metric_name, v, step=i)
        
        return ret_map

    def get_params(self, infer_types: bool = False) -> ObjectView:
        """Return model parameters.

        Args:
            infer_types: If True, will try to infer value types since MLFlow 
                        stores them as strings.
        
        Returns:
            ObjectView containing the model parameters.
        """
        ret = ObjectView(self.run.data.params)
        if infer_types:
            for k, v in ret.items():
                ret[k] = yaml.safe_load(v)
        return ret

    def get_run_id(self) -> str:
        """Return the experiment's run ID.
        
        Returns:
            The MLFlow run ID.
        """
        return self.run_id

    def get_metrics(self, as_dict: bool = False) -> Union[pd.DataFrame, ObjectView]:
        """Collect all metrics available for a given run.

        Args:
            as_dict: If True, return metrics as a dict where vector metrics 
                    are returned as numpy arrays. If False, return as DataFrame.

        Returns:
            Either a pandas DataFrame with all metrics or an ObjectView dict 
            with metrics as numpy arrays for vector metrics.
        """
        # Collecting all metrics in a dataframe
        df = pd.DataFrame(columns=['metric', 'step', 'value'])
        for metric_name in self.run.data.metrics.keys():
            for m in self.run_client.get_metric_history(self.run_id, metric_name):
                df.loc[len(df)] = metric_name, m.step, m.value
        df.sort_values(['metric', 'step'], inplace=True, ignore_index=True)
        
        if not as_dict: 
            return df
    
        # Creating a dataframe where vectorized metrics are saved as lists
        grp = df.groupby('metric').value.agg(lambda x: x.iloc[0] if len(x) == 1 else x.to_list()).to_dict()
        
        # Lists are converted to np.arrays
        for k, v in grp.items():
            if hasattr(v, '__iter__'):
                grp[k] = np.array(v)
        return ObjectView(grp)

    def get_dataframe(self, var_name: str, folder: str = '') -> pd.DataFrame:
        """Load a pandas DataFrame saved to MLFlow as a parquet file.

        Args:
            var_name: The name the dataframe has within MLFlow.
            folder: The path (in MLFlow) from where the dataframe will be loaded.
        
        Returns:
            pandas DataFrame with the loaded data.
        """
        return self.get_artifact(var_name + '.parquet', folder, pd.read_parquet)

    def get_numpy(self, var_name: str, folder: str = '') -> np.ndarray:
        """Load a numpy array saved to MLFlow as a .npy file.

        Args:
            var_name: The name the array has within MLFlow.
            folder: The path (in MLFlow) from where the array will be loaded.
        
        Returns:
            numpy ndarray with the loaded data.
        """
        return self.get_artifact(var_name + '.npy', folder, np.load, allow_pickle=False)

    def get_artifact(self, var_name: str, folder: str = '', load_func: Callable = np.load, 
                     **load_func_kwargs: Any) -> Any:
        """Load an artifact using the provided loading function.

        Args:
            var_name: The name the artifact has within MLFlow (must include extensions).
            folder: The path (in MLFlow) from where the artifact will be loaded.
            load_func: The function used to load the required artifact.
            **load_func_kwargs: Additional parameters to be passed to load_func.
        
        Returns:
            Whatever the load_func returns.
        """
        with tempfile.TemporaryDirectory() as temp_path:
            full_path = os.path.join(folder, var_name)
            local_path = mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=full_path, dst_path=temp_path)
            return load_func(local_path, **load_func_kwargs)

    def get_experiment(self) -> mlflow.entities.Experiment:
        """Return the experiment that contains the MLFlow run.
        
        Returns:
            MLFlow Experiment instance containing this run.
        """
        exp_id = self.run.info.experiment_id
        return mlflow.get_experiment(exp_id)

    def log_pipeline(self, pipeline: Pipeline, title: str, folder: str = '', 
                     metadata: Optional[Dict[str, str]] = None) -> None:
        """Log a scikit-learn pipeline as an HTML visualization.

        Args:
            pipeline: The scikit-learn Pipeline to visualize.
            title: Title for the HTML page.
            folder: The path (in MLFlow) where the HTML file will be saved.
            metadata: Optional dictionary of metadata to include in the HTML.
        """
        # Enable diagram display
        sklearn.set_config(display='diagram')

        # Generate HTML representation of pipeline
        pipeline_html = estimator_html_repr(pipeline)

        # Save to temporary file with specific name
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_viz_file = os.path.join(temp_dir, 'model_pipeline.html')

            meta_html = '\n'.join([f'<p><strong>{key}:</strong> {value}</p>' for key, value in metadata.items()]) if metadata else ''

            with open(pipeline_viz_file, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>SOM Pipeline - {self.run.info.run_id[:8]}</title>
                        <style>
                            body {{ 
                                background-color: #1e1e1e; 
                                color: #d4d4d4;
                                padding: 20px;
                                font-family: monospace;
                            }}
                            h1 {{ color: #4fc3f7; }}
                            .metadata {{ 
                                background: #2d2d2d; 
                                padding: 15px; 
                                border-radius: 8px;
                                margin: 20px 0;
                            }}
                        </style>
                    </head>
                    <body>
                        <h1>🔧 {title}</h1>
                        <div class="metadata">
                            <p><strong>Run ID:</strong> {self.run.info.run_id}</p>
                            {meta_html}
                        </div>
                        <h2>Pipeline Structure</h2>
                        {pipeline_html}
                    </body>
                    </html>
                """)

            mlflow.log_artifact(pipeline_viz_file, artifact_path=folder)
