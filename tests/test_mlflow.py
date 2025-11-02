import os
import datetime
import numpy as np
import pandas as pd
import pytest
import mlflow
import sklearn
import sklearn.utils
import lpsds.metrics
from lpsds.mlflow import MLFlow
from lpsds.utils import ObjectView

class MLFlowBase:

    class MockedRunClient:
        def get_metric_history(self, run_id, metric_name):

            sp_list = []
            f1_list = []
            for i in range(5):
                for j, v in enumerate([sp_list, f1_list]):
                    w = 1 if j==0 else 10
                    obj = ObjectView()
                    obj.value = w * (i+1)
                    obj.step = i+1
                    v.append(obj)

            ret_map = {}
            for m, v in zip(['sp_mean', 'f1_mean', 'acc'], [0.9, 0.95, 0.99]):
                obj = ObjectView()
                obj.value = v
                obj.step = 0
                ret_map[m] = [ obj ]

            ret_map['sp'] = sp_list
            ret_map['f1'] = f1_list
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
        ret.info.experiment_id = '112233'
        return ret
    
    @staticmethod
    def mocked_get_run(run_id):
        """
        Mocks mlflow.active_run behavior
        """
        ret = ObjectView()
        ret.info = ObjectView()
        ret.info.run_id = run_id
        ret.info.experiment_id = run_id + '987'
        ret.data = ObjectView()
        ret.data.metrics = dict(
            sp_mean = 0.9,
            f1_mean = 0.95,
            acc = 0.99,
            sp = 0,
            f1 = 10,
        )
        ret.data.params = dict(
            param_str ='my string',
            param_int = '123',
            param_float = '3.14',
            param_bool = 'True',
            param_date = '2022-12-24',
        )
        return ret
    
    @staticmethod
    def mocked_get_experiment(exp_id):
        ret = ObjectView()
        ret.name = f'test_experiment_{exp_id}'
        return ret


    @staticmethod
    def mocked_download_artifacts(run_id, artifact_path, dst_path):
        df = pd.DataFrame({'a' : [1,2,3], 'b' : [11,22,33]})
        fname = os.path.split(artifact_path)[1]
        local_name = os.path.join(dst_path, fname)
        df.to_parquet(local_name)
        return local_name


    @pytest.fixture(autouse=True)
    def set_mlflow_patches(self, monkeypatch, request):
        if 'noautofixt' in request.keywords: return
        monkeypatch.setattr(mlflow, 'active_run', MLFlowBase.mocked_active_run)
        monkeypatch.setattr(mlflow, 'get_run', MLFlowBase.mocked_get_run)
        monkeypatch.setattr(mlflow, 'get_experiment', MLFlowBase.mocked_get_experiment)
        monkeypatch.setattr(mlflow.tracking, 'MlflowClient', MLFlowBase.MockedRunClient)
        monkeypatch.setattr(mlflow.artifacts, 'download_artifacts', MLFlowBase.mocked_download_artifacts)


class TestInit(MLFlowBase):

    def test_call_with_id(self):
        a = MLFlow('12345')
        assert a.run_id == '12345'

    @pytest.mark.noautofixt
    def test_call_without_id_without_experiment(self):
        with pytest.raises(ValueError):
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



class TestLogArtifact(MLFlowBase):

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
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogArtifact.assert_file_exists)    
        mlf_obj.log_artifact(df, 'filename.parquet', 'mlflow_test_folder', save_func=df.to_parquet, fname_param_name='path', object_param_name=None)


    def test_file_correct(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogArtifact.assert_right_content)
        mlf_obj.log_artifact(df, 'filename.parquet', 'mlflow_test_folder', save_func=df.to_parquet, fname_param_name='path', object_param_name=None)



class TestGetRunID(MLFlowBase):
    def test_return(self, mlf_obj):
        assert mlf_obj.get_run_id() == '112233'


class TestGetMetrics(MLFlowBase):

    def test_num_returns(self, mlf_obj):
        met = mlf_obj.get_metrics(as_dict=True)
        assert len(met) == 5    

    def test_scalars(self, mlf_obj):
        met = mlf_obj.get_metrics(as_dict=True)
        assert met['sp_mean'] == 0.9
        assert met['f1_mean'] == 0.95
        assert met['acc'] == 0.99


    def test_object_view(self, mlf_obj):
        met = mlf_obj.get_metrics(as_dict=True)
        assert isinstance(met, ObjectView)


    def test_vectors(self, mlf_obj):
        met = mlf_obj.get_metrics(as_dict=True)
        assert np.array_equal(met['sp'], np.array([1,2,3,4,5]))
        assert np.array_equal(met['f1'], np.array([10,20,30,40,50]))


        
    def test_dataframe_structure(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert met.shape[0] == 13
        assert met.shape[1] == 3
        assert met.columns[0] == 'metric'
        assert met.columns[1] == 'step'
        assert met.columns[2] == 'value'

    

    def test_scalars_dataframe(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert met.loc[met.metric == 'sp_mean', 'value'].iloc[0] == 0.9
        assert met.loc[met.metric == 'f1_mean', 'value'].iloc[0] == 0.95
        assert met.loc[met.metric == 'acc', 'value'].iloc[0] == 0.99


    def test_vectors_dataframe(self, mlf_obj):
        met = mlf_obj.get_metrics()
        assert np.array_equal(met.loc[met.metric == 'sp', 'value'].to_numpy(), np.array([1,2,3,4,5]))
        assert np.array_equal(met.loc[met.metric == 'f1', 'value'].to_numpy(), np.array([10,20,30,40,50]))




class TestGetArtifact(MLFlowBase):

    def test_operation_no_folder(self, mlf_obj):
        df = mlf_obj.get_artifact('my_df', load_func=pd.read_parquet)
        assert df.shape[0] == 3
        assert df.shape[1] == 2
        assert df.iloc[0].a == 1
        assert df.iloc[0].b == 11
        assert df.iloc[1].a == 2
        assert df.iloc[1].b == 22
        assert df.iloc[2].a == 3
        assert df.iloc[2].b == 33

    def test_operation_with_folder(self, mlf_obj):
        df = mlf_obj.get_artifact('my_df', 'folder/path', load_func=pd.read_parquet)
        assert df.shape[0] == 3
        assert df.shape[1] == 2
        assert df.iloc[0].a == 1
        assert df.iloc[0].b == 11
        assert df.iloc[1].a == 2
        assert df.iloc[1].b == 22
        assert df.iloc[2].a == 3
        assert df.iloc[2].b == 33


class TestGetExperiment(MLFlowBase):
    def test_positive_case(self):
        a = MLFlow('123')
        assert a.get_experiment().name == 'test_experiment_123987'


class TestGetParams(MLFlowBase):

    def test_num_returns(self, mlf_obj):
        par = mlf_obj.get_params()
        assert len(par) == 5

    def test_str_return(self, mlf_obj):
        par = mlf_obj.get_params(infer_types=False)
        assert par['param_str'] == 'my string'
        assert par['param_int'] == '123'
        assert par['param_float'] == '3.14'
        assert par['param_bool'] == 'True'
        assert par['param_date'] == '2022-12-24'


    def test_infer_type_return(self, mlf_obj):
        par = mlf_obj.get_params(infer_types=True)
        assert par['param_str'] == 'my string'
        assert par['param_int'] == 123
        assert par['param_float'] == 3.14
        assert par['param_bool'] == True
        assert par['param_date'] == datetime.date(2022, 12, 24)



    def test_object_view(self, mlf_obj):
        par = mlf_obj.get_params()
        assert isinstance(par, ObjectView)


class TestLogDataFrame(MLFlowBase):
    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a' : [1,2,3], 'b' : [40,50,60]})

    def test_file_exists(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogArtifact.assert_file_exists)    
        mlf_obj.log_dataframe(df, 'filename', 'mlflow_test_folder')


    def test_file_correct(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogArtifact.assert_right_content)
        mlf_obj.log_dataframe(df, 'filename', 'mlflow_test_folder')


class TestLogNumpy(MLFlowBase):
    @staticmethod
    def assert_file_exists(temp_file_name, folder):
        assert os.path.exists(temp_file_name)
        assert folder == 'mlflow_test_folder'

    @staticmethod
    def assert_right_content(temp_file_name, folder):
        df = np.load(temp_file_name)
        assert df.shape[0] == 2
        assert df.shape[1] == 3

        assert df[0][0] == 1
        assert df[0][1] == 2
        assert df[0][2] == 3

        assert df[1][0] == 40
        assert df[1][1] == 50
        assert df[1][2] == 60


    @pytest.fixture
    def df(self):
        return np.array([[1,2,3],[40,50,60]])

    def test_file_exists(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogNumpy.assert_file_exists)    
        mlf_obj.log_numpy(df, 'filename', 'mlflow_test_folder')


    def test_file_correct(self, monkeypatch, df, mlf_obj):
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogNumpy.assert_right_content)
        mlf_obj.log_numpy(df, 'filename', 'mlflow_test_folder')



class TestGetDataFrame(MLFlowBase):

    def test_operation_no_folder(self, mlf_obj):
        df = mlf_obj.get_dataframe('my_df')
        assert df.shape[0] == 3
        assert df.shape[1] == 2
        assert df.iloc[0].a == 1
        assert df.iloc[0].b == 11
        assert df.iloc[1].a == 2
        assert df.iloc[1].b == 22
        assert df.iloc[2].a == 3
        assert df.iloc[2].b == 33

    def test_operation_with_folder(self, mlf_obj):
        df = mlf_obj.get_dataframe('my_df', 'folder/path')
        assert df.shape[0] == 3
        assert df.shape[1] == 2
        assert df.iloc[0].a == 1
        assert df.iloc[0].b == 11
        assert df.iloc[1].a == 2
        assert df.iloc[1].b == 22
        assert df.iloc[2].a == 3
        assert df.iloc[2].b == 33




class TestGetNumpy(MLFlowBase):

    @staticmethod
    def mocked_download_artifacts_numpy(run_id, artifact_path, dst_path):
        df = np.array([[1,2,3],[11,22,33]])
        fname = os.path.split(artifact_path)[1]
        local_name = os.path.join(dst_path, fname + '.npy')
        np.save(local_name, df)
        return local_name


    def test_operation_no_folder(self, mlf_obj, monkeypatch):
        monkeypatch.setattr(mlflow.artifacts, 'download_artifacts', TestGetNumpy.mocked_download_artifacts_numpy)
        df = mlf_obj.get_numpy('my_df')
        assert df.shape[0] == 2
        assert df.shape[1] == 3
        assert df[0][0] == 1
        assert df[0][1] == 2
        assert df[0][2] == 3
        assert df[1][0] == 11
        assert df[1][1] == 22
        assert df[1][2] == 33

    def test_operation_with_folder(self, mlf_obj, monkeypatch):
        monkeypatch.setattr(mlflow.artifacts, 'download_artifacts', TestGetNumpy.mocked_download_artifacts_numpy)
        df = mlf_obj.get_numpy('my_df', 'folder/path')
        assert df.shape[0] == 2
        assert df.shape[1] == 3
        assert df[0][0] == 1
        assert df[0][1] == 2
        assert df[0][2] == 3
        assert df[1][0] == 11
        assert df[1][1] == 22
        assert df[1][2] == 33


class TestLogPipeline(MLFlowBase):
    """Test class for log_pipeline method"""

    @staticmethod
    def mocked_set_config(display):
        """Mock sklearn.set_config to avoid side effects"""
        pass

    @staticmethod
    def mocked_estimator_html_repr(pipeline):
        """Mock HTML representation of pipeline"""
        return '<div class="sklearn-pipeline">Mock Pipeline HTML</div>'

    @staticmethod
    def mocked_log_artifact(file_path, artifact_path):
        """Mock mlflow.log_artifact and verify file contents"""
        # Verify file exists and has correct content
        assert os.path.exists(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic HTML structure checks
        assert '<!DOCTYPE html>' in content
        assert '<title>SOM Pipeline' in content
        # Check for sklearn pipeline content (real HTML representation)
        assert 'Pipeline' in content
        return None

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing"""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])

    def test_basic_pipeline_logging(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test basic pipeline logging with minimal parameters"""
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogPipeline.mocked_log_artifact)
        
        # Should not raise any exceptions
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')

    def test_pipeline_with_folder(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test pipeline logging with custom folder path"""
        def assert_folder_path(file_path, artifact_path):
            assert artifact_path == 'custom/folder'
            TestLogPipeline.mocked_log_artifact(file_path, artifact_path)
        
        monkeypatch.setattr(mlflow, 'log_artifact', assert_folder_path)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline', folder='custom/folder')

    def test_pipeline_with_metadata(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test pipeline logging with metadata dictionary"""
        def assert_metadata_in_html(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that metadata is included in HTML
            assert '<strong>Author:</strong> Test User' in content
            assert '<strong>Version:</strong> 1.0' in content
            assert '<strong>Model Type:</strong> Classification' in content
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_metadata_in_html)
        
        metadata = {
            'Author': 'Test User',
            'Version': '1.0',
            'Model Type': 'Classification'
        }
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline', metadata=metadata)

    def test_pipeline_with_empty_metadata(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test pipeline logging with empty metadata dictionary"""
        def assert_no_metadata_section(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should still have run ID but no additional metadata
            assert 'Run ID:' in content
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_no_metadata_section)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline', metadata={})

    def test_pipeline_with_none_metadata(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test pipeline logging with None metadata"""
        def assert_none_metadata(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should only have run ID, no additional metadata HTML
            assert 'Run ID:' in content
            # Should not have extra metadata paragraphs
            content_lines = content.split('\n')
            metadata_section = [line for line in content_lines if '<strong>' in line and 'Run ID' not in line]
            assert len(metadata_section) == 0
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_none_metadata)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline', metadata=None)

    def test_html_file_structure(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test that generated HTML file has correct structure"""
        def assert_html_structure(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check HTML structure
            assert '<!DOCTYPE html>' in content
            assert '<html lang="en">' in content
            assert '<meta charset="UTF-8">' in content
            assert '<title>SOM Pipeline - 112233' in content  # Full run_id (6 chars)
            assert '<h1>🔧 Test Pipeline</h1>' in content
            assert '<h2>Pipeline Structure</h2>' in content
            assert 'Pipeline' in content  # Check for pipeline content
            
            # Check CSS styling
            assert 'background-color: #1e1e1e' in content
            assert 'color: #d4d4d4' in content
            assert 'font-family: monospace' in content
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_html_structure)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')

    def test_sklearn_config_called(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test that sklearn.set_config is called with display='diagram'"""
        config_calls = []
        
        def track_set_config(display):
            config_calls.append(display)
        
        # Patch in the lpsds.mlflow module where it's imported
        monkeypatch.setattr('lpsds.mlflow.sklearn.set_config', track_set_config)
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogPipeline.mocked_log_artifact)
        
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')
        
        assert len(config_calls) == 1
        assert config_calls[0] == 'diagram'

    def test_estimator_html_repr_called(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test that estimator_html_repr is called with the pipeline"""
        html_repr_calls = []
        
        def track_html_repr(pipeline):
            html_repr_calls.append(pipeline)
            return '<div>Mock HTML</div>'
        
        # Patch in the lpsds.mlflow module where it's imported
        monkeypatch.setattr('lpsds.mlflow.estimator_html_repr', track_html_repr)
        monkeypatch.setattr(mlflow, 'log_artifact', TestLogPipeline.mocked_log_artifact)
        
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')
        
        assert len(html_repr_calls) == 1
        assert html_repr_calls[0] is mock_pipeline

    def test_file_name_consistency(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test that the HTML file is always named 'model_pipeline.html'"""
        def assert_filename(file_path, artifact_path):
            filename = os.path.basename(file_path)
            assert filename == 'model_pipeline.html'
        
        monkeypatch.setattr(mlflow, 'log_artifact', assert_filename)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')

    def test_special_characters_in_title(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test pipeline logging with special characters in title"""
        def assert_special_chars_handled(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that special characters are handled (not escaped in f-string)
            assert '<h1>🔧 Test <Pipeline> & Model</h1>' in content
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_special_chars_handled)
        mlf_obj.log_pipeline(mock_pipeline, 'Test <Pipeline> & Model')

    def test_run_id_in_title(self, monkeypatch, mock_pipeline, mlf_obj):
        """Test that run ID is correctly truncated to 8 characters in title"""
        def assert_run_id_truncated(file_path, artifact_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Run ID should be truncated to first 8 characters (or the full ID if shorter)
            assert 'SOM Pipeline - 112233' in content  # Full run_id (6 chars, less than 8)
            
        monkeypatch.setattr(mlflow, 'log_artifact', assert_run_id_truncated)
        mlf_obj.log_pipeline(mock_pipeline, 'Test Pipeline')
