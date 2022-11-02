import os
import numpy as np
import pandas as pd
import pytest
import mlflow
import lpsds.metrics
from lpsds.mlflow import log_statistics, log_dataframe

class TestLogStatistics:

    @staticmethod
    def mocked_log_metrics(metrics_map):
        """
        Mocks mlflow.log_metrics behavior
        """
        return None

    @staticmethod
    def mocked_bootstrap_estimate(vec, ci=95, n_boot=1000, seed=None):
        """
        Returns a mocked result for bootstrap_estimate
        """
        return vec.mean(), vec.mean()-vec.min(), vec.mean()+vec.max()

    @pytest.fixture(autouse=True)
    def set_patches(self, monkeypatch):
        monkeypatch.setattr(lpsds.metrics, 'bootstrap_estimate', TestLogStatistics.mocked_bootstrap_estimate)
        monkeypatch.setattr(mlflow, 'log_metrics', TestLogStatistics.mocked_log_metrics)

    @pytest.fixture
    def cv_map(self):
        """
        Returns the mocked data for each test
        """        
        return {
            'test_sp' : np.array([1,2,3,4,5]), #mean = 3, err_min=2, err_max=8
            'estimators' : np.array([1,2,3,4,5]),
            'test_auc' : np.array([10,2,30,4,50]), #mean=19.2, err_min=17.2, err_max=69.2
        }

    def test_only_2_returned(self, cv_map):
        """
        Only 2 metrics must be returned
        """
        ret = log_statistics(cv_map)
        assert len(ret) == 2


    def test_metric_names(self, cv_map):
        """
        Only SP and AUC must be returned
        """
        ret = log_statistics(cv_map)
        assert 'sp' in ret
        assert 'auc' in ret

    def test_metric_values(self, cv_map):
        """
        Only metric values must be correct
        """
        ret = log_statistics(cv_map)
        assert ret['sp']['sp_mean'] == 3
        assert ret['sp']['sp_err_min'] == 2
        assert ret['sp']['sp_err_max'] == 8

        assert ret['auc']['auc_mean'] == 19.2
        assert ret['auc']['auc_err_min'] == 17.2
        assert ret['auc']['auc_err_max'] == 69.2


class TestLogDataFrame():

    @staticmethod
    def assert_file_exists(temp_file_name, folder):
        assert os.path.exists(temp_file_name)
        assert folder == 'mlflow_test_folder'

    @staticmethod
    def assert_right_content(temp_file_name, folder):
        df = pd.read_parquet(temp_file_name)
        assert df.shape[0] == 3
        assert df.shape[1] == 2

        assert df.a.iloc[0] == 1
        assert df.a.iloc[1] == 2
        assert df.a.iloc[2] == 3

        assert df.b.iloc[0] == 40
        assert df.b.iloc[1] == 50
        assert df.b.iloc[2] == 60


    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a' : [1,2,3], 'b' : [40,50,60]})

    def test_file_exists(self, monkeypatch, df):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogDataFrame.assert_file_exists)        
        log_dataframe(df, 'filename.parquet', 'mlflow_test_folder')


    def test_file_correct(self, monkeypatch, df):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogDataFrame.assert_right_content)
        log_dataframe(df, 'filename.parquet', 'mlflow_test_folder')