import os
import numpy as np
import pandas as pd
import pytest
import mlflow
import lpsds.metrics
from lpsds.mlflow import MLFlow
from lpsds.utils import ObjectView

class MLFlowBase:

    class MockedRunClient:
        def get_metric_history(self, run_id, metric_name):

            sp_list = []
            f1_list = []
            for i in range(5):
                sp_obj = ObjectView()
                sp_obj.value = i+1
                f1_obj = ObjectView()
                f1_obj.value = (i+1) * 10
                sp_list.append(sp_obj)
                f1_list.append(f1_obj)

            ret_map = {
                'sp' : sp_list,
                'f1' : f1_list,
            }
            return ret_map[metric_name]

    @pytest.fixture
    def mlf_obj(self):
        return MLFlow()

    @staticmethod
    def mocked_active_run():
        """
        Mocks mlflow.active_run behavior
        """
        ret = ObjectView()
        ret.info = ObjectView()
        ret.info.run_id = '112233'
        return ret
    
    @staticmethod
    def mocked_get_run(run_id):
        """
        Mocks mlflow.active_run behavior
        """
        ret = ObjectView()
        ret.data = ObjectView()
        ret.data.metrics = dict(
            sp_mean = 0.9,
            f1_mean = 0.95,
            acc = 0.99
        )
        return ret

    @pytest.fixture(autouse=True)
    def set_mlflow_patches(self, monkeypatch, request):
        if 'noautofixt' in request.keywords: return
        monkeypatch.setattr(mlflow, 'active_run', MLFlowBase.mocked_active_run)
        monkeypatch.setattr(mlflow, 'get_run', MLFlowBase.mocked_get_run)
        monkeypatch.setattr(mlflow.tracking, 'MlflowClient', MLFlowBase.MockedRunClient)


class TestInit(MLFlowBase):

    def test_call_with_id(self):
        a = MLFlow('12345')
        assert a.run_id == '12345'

    @pytest.mark.noautofixt
    def test_call_without_id_without_experiment(self):
        with pytest. raises(ValueError):
            MLFlow()


    def test_call_without_id(self):
        a = MLFlow()
        assert a.run_id == '112233'


class TestLogStatistics(MLFlowBase):

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
    
    @staticmethod
    def mocked_log_metric(key, value, step):
        return None

    @pytest.fixture(autouse=True)
    def set_patches(self, monkeypatch):
        monkeypatch.setattr(lpsds.metrics, 'bootstrap_estimate', TestLogStatistics.mocked_bootstrap_estimate)
        monkeypatch.setattr(mlflow, 'log_metrics', TestLogStatistics.mocked_log_metrics)
        monkeypatch.setattr(mlflow, 'log_metric', TestLogStatistics.mocked_log_metric)

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

    def test_only_2_returned(self, cv_map, mlf_obj):
        """
        Only 2 metrics must be returned
        """
        ret = mlf_obj.log_statistics(cv_map)
        assert len(ret) == 2


    def test_metric_names(self, cv_map, mlf_obj):
        """
        Only SP and AUC must be returned
        """
        ret = mlf_obj.log_statistics(cv_map)
        assert 'sp' in ret
        assert 'auc' in ret

    def test_metric_values(self, cv_map, mlf_obj):
        """
        Only metric values must be correct
        """
        ret = mlf_obj.log_statistics(cv_map)
        assert ret['sp']['sp_mean'] == 3
        assert ret['sp']['sp_err_min'] == 2
        assert ret['sp']['sp_err_max'] == 8

        assert ret['auc']['auc_mean'] == 19.2
        assert ret['auc']['auc_err_min'] == 17.2
        assert ret['auc']['auc_err_max'] == 69.2
    
    @staticmethod
    def assert_metrics_logged(metric_map):
        assert metric_map['sp_mean'] == 3
        assert metric_map['sp_err_min'] == 2
        assert metric_map['sp_err_max'] == 8

    def test_mlflow_submission(self, monkeypatch, mlf_obj):
        met_map = {'test_sp' : np.array([1,2,3,4,5])} #mean = 3, err_min=2, err_max=8
        monkeypatch.setattr(mlflow, 'log_metrics', TestLogStatistics.assert_metrics_logged)
        mlf_obj.log_statistics(met_map)
    

    def test_metrics_vector_logged(self, monkeypatch, mlf_obj):
        self.metric_list = []
        def mocked_log_metric(key, value, step):
            self.metric_list.append((key, value, step))


        met_map = {'test_sp' : np.array([1,2,3,4,5])} #mean = 3, err_min=2, err_max=8
        monkeypatch.setattr(mlflow, 'log_metric', mocked_log_metric)
        mlf_obj.log_statistics(met_map)
        for i, items in enumerate(self.metric_list):
            k,v,s = items
            assert k == 'sp'
            assert v == i+1
            assert s == i



class TestLogDataFrame(MLFlowBase):

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

    def test_file_exists(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogDataFrame.assert_file_exists)        
        mlf_obj.log_dataframe(df, 'filename.parquet', 'mlflow_test_folder')


    def test_file_correct(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogDataFrame.assert_right_content)
        mlf_obj.log_dataframe(df, 'filename.parquet', 'mlflow_test_folder')



class TestGetRunID(MLFlowBase):
    def test_return(self, mlf_obj):
        assert mlf_obj.get_run_id() == '112233'


class TestGetMetrics(MLFlowBase):

    def test_num_returns(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert len(met) == 5
    

    def test_scalars(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert met['sp_mean'] == 0.9
        assert met['f1_mean'] == 0.95
        assert met['acc'] == 0.99


    def test_vectors(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert np.array_equal(met['sp'], np.array([1,2,3,4,5]))
        assert np.array_equal(met['f1'], np.array([10,20,30,40,50]))


#    class TestGetDataFrame(MLFlowBase):

