import mlflow

def log_dataframe(df, var_name, mlflow_folder=''):
    with tempfile.TemporaryDirectory() as temp_path:
        temp_file_name = os.path.join(temp_path, var_name + '.parquet')
        df.to_parquet(temp_file_name, index=False)
        mlflow.log_artifact(temp_file_name, mlflow_folder)
