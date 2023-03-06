"""model module tests"""

import pytest
import pandas as pd
import numpy as np
from lpsds.model import get_operation_model, create_validation_dataset, feature_importances


class DummyModel:
    """Dummy model for tests"""
    def __init__(self,fake_output, id):
        """constructor"""
        self.fake_output = fake_output
        self.id = id

    def predict(self, X):
        """predict emulator"""
        ret = np.ones(self.fake_output.shape)
        #Cutting in zero
        ret[self.fake_output<0] = -1.
        return ret

    def predict_proba(self, X):
        """predict_proba emulator"""
        return self.fake_output


class TestGetOperationModel:
    """Tests get_operation_model"""

    @pytest.fixture
    def X(self):
        """Fake X"""
        return np.random.randn(5,2)
    
    @pytest.fixture
    def y_true(self):
        """Fake y"""
        return np.array([0.2, -1, 0, -0.5, 0.9])
    
    @pytest.fixture
    def models(self):
        """Dummy models"""
        mod1 = DummyModel(np.array([0.2, +1, 0, -0.5, -0.9]), id='model1') #2 wrongs (+1 and -0.9)
        mod2 = DummyModel(np.array([0.2, -1, -0.1, -0.5, 0.9]), id='model2') #1 wrongs 9-0.1
        mod3 = DummyModel(np.array([0.2, +1, 0, +0.5, -0.9]), id='model3') #3 wrongs (+1, +0.5 and -0.9)
        return {'estimator' : [mod1, mod2, mod3]}

    @pytest.fixture(autouse=True)
    def global_var(self):
        pytest.iteration_number = 0

    def dummy_metric(self, y_pred, y_true, **kwargs):
        """
        Dummy metric function.
        """
        mod_idx_to_return = kwargs.get('mod_idx', 0)
        ret = -1 if pytest.iteration_number == mod_idx_to_return else +1
        pytest.iteration_number += 1
        return ret

    
    def test_default_case(self, models, X, y_true):
        """Test behavior with default values"""
        mod = get_operation_model(models, X, y_true)
        assert mod.id == 'model2'

    def test_metric_override(self, models, X, y_true):
        """Test if overriding metrics is working"""
        self.metric_idx = 0
        mod = get_operation_model(models, X, y_true, metric=self.dummy_metric)
        assert mod.id == 'model1'

    def test_metrics_kwargs(self, models, X, y_true):
        """Test kwargs for metric function"""
        self.metric_idx = 0
        mod = get_operation_model(models, X, y_true, metric=self.dummy_metric, mod_idx=2)
        assert mod.id == 'model3'



class TesteCreateValidationDataset:
    """Tests create_validation_dataset"""

    @pytest.fixture
    def X(self):
        """Fake X"""
        return pd.DataFrame({
            'a' : [11, 21],
            'b' : [12, 22],
            })
    
    @pytest.fixture
    def y_true(self):
        """Fake y"""
        return np.array([0.2, -1])

    @pytest.fixture
    def y_pred(self):
        """Fake y"""
        return np.array([0.1, -0.8])

    @pytest.fixture
    def model(self, y_pred):
        """Dummy models"""
        return DummyModel(y_pred, id='model')


    def test_columns_order(self, model, X, y_true):
        """Test whether the returned columns are OK"""
        ret = create_validation_dataset(model, X, y_true, proba=False)
        assert ret.columns[0] == 'a'
        assert ret.columns[1] == 'b'
        assert ret.columns[2] == 'y_true'
        assert ret.columns[3] == 'y_pred'

    def test_values(self, model, X, y_true):
        """Test whether the returned values are OK"""
        ret = create_validation_dataset(model, X, y_true, proba=False)
        assert ret.a.iloc[0] == 11
        assert ret.a.iloc[1] == 21
        assert ret.b.iloc[0] == 12
        assert ret.b.iloc[1] == 22
        assert ret.y_true.iloc[0] == 0.2
        assert ret.y_true.iloc[1] == -1
        assert ret.y_pred.iloc[0] == +1
        assert ret.y_pred.iloc[1] == -1


    def test_predict_proba(self, model, X, y_true):
        """Tests predic probability case"""
        ret = create_validation_dataset(model, X, y_true, proba=True)
        assert ret.a.iloc[0] == 11
        assert ret.a.iloc[1] == 21
        assert ret.b.iloc[0] == 12
        assert ret.b.iloc[1] == 22
        assert ret.y_true.iloc[0] == 0.2
        assert ret.y_true.iloc[1] == -1
        assert ret.y_pred.iloc[0] == 0.1
        assert ret.y_pred.iloc[1] == -0.8

    def test_copy_true(self, model, X, y_true):
        """Test copy = True case for X and y"""
        create_validation_dataset(model, X, y_true, copy=True)
        assert 'y_true' not in X.columns
        assert 'y_pred' not in X.columns
        assert X.a.iloc[0] == 11
        assert X.a.iloc[1] == 21
        assert X.b.iloc[0] == 12
        assert X.b.iloc[1] == 22

    def test_copy_false(self, model, X, y_true):
        """Test copy = False case for X and y"""
        create_validation_dataset(model, X, y_true, copy=False)
        assert X.a.iloc[0] == 11
        assert X.a.iloc[1] == 21
        assert X.b.iloc[0] == 12
        assert X.b.iloc[1] == 22
        assert X.y_true.iloc[0] == 0.2
        assert X.y_true.iloc[1] == -1
        assert X.y_pred.iloc[0] == 0.1
        assert X.y_pred.iloc[1] == -0.8


class TestFeatureImportance:
    
    class Model:
        def predict(self, X):
            ret = X.sum(axis=1)
            return ret if isinstance(ret, np.ndarray) else ret.to_numpy()

    @pytest.fixture
    def model(self):
        return TestFeatureImportance.Model()
    
    @pytest.fixture
    def X(self):
        return pd.DataFrame({'a': [3, 11], 'b' : [20, 30]})
    
    @pytest.fixture
    def y_true(self):
        return pd.Series([22, 45])
    
    def test_columns(self, model, X, y_true):
        relev = feature_importances(model, X, y_true)
        assert relev.index.name == 'feature'
        assert relev.columns[0] == 'importance'

    def test_rows(self, model, X, y_true):
        relev = feature_importances(model, X, y_true)
        assert relev.index[0] == 'a'
        assert relev.index[1] == 'b'

    def test_shape(self, model, X, y_true):
        relev = feature_importances(model, X, y_true)
        assert relev.shape[0] == 2
        assert relev.shape[1] == 1

    def test_positive_case(self, model, X, y_true):
        relev = feature_importances(model, X, y_true)
        assert relev.loc['a', 'importance'] == -36
        assert relev.loc['b', 'importance'] == -50

    @pytest.mark.parametrize(("x", "y"), [ 
                                (pd.DataFrame({'a': [3, 11], 'b' : [20, 30]}),            pd.Series([22, 45])),
                                (pd.DataFrame({'a': [3, 11], 'b' : [20, 30]}),            pd.Series([22, 45]).to_numpy()),
                                (pd.DataFrame({'a': [3, 11], 'b' : [20, 30]}).to_numpy(), pd.Series([22, 45])),
                                (pd.DataFrame({'a': [3, 11], 'b' : [20, 30]}).to_numpy(), pd.Series([22, 45]).to_numpy()),
                            ])
    def test_type_combinations(self, model, x,y):
        relev = feature_importances(model, x, y)
        assert relev.iloc[0].importance == -36
        assert relev.iloc[1].importance == -50