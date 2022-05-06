"""model module tests"""

import pytest
import numpy as np
from model import get_operation_model


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
    def models(self, y_true):
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

    def test_default_case(self):
        """Test behavior with default values"""
        pass

    def test_predict_proba(self):
        """Tests predic probability case"""
        pass

    def test_copy_true(self):
        """Test copy = True case for X and y"""
        pass

    def test_copy_false(self):
        """Test copy = False case for X and y"""
        pass