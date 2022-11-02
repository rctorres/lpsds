import pytest
import numpy as np
import pandas as pd
from lpsds.operation import get_operation_model


class TestGetOperationModel:

    @pytest.fixture
    def cv_model(self):
        return dict(
            test_accuracy=np.array([3,9,6,8,4]), #max ids = 1
            test_mse=np.array([3,9,6,2,4]), #min idx = 3
            estimator=['model_40', 'model_41', 'model_42', 'model_43', 'model_44']
        )
    
    @pytest.fixture
    def cv_splits(self):
        return [
            ([2,3,4,5,6,7,8,9], [0,1]), #fold 0
            ([0,1,4,5,6,7,8,9], [2,3]), #fold 1
            ([0,1,2,3,6,7,8,9], [4,5]), #fold 2
            ([0,1,2,3,4,5,8,9], [6,7]), #fold 3
            ([0,1,2,3,4,5,6,7], [8,9]), #fold 4
        ]
    
    @pytest.fixture
    def X(self):
        ret = pd.DataFrame(columns=['x1', 'x2'])
        ret.loc[len(ret)] = 10,20
        ret.loc[len(ret)] = 11,21
        ret.loc[len(ret)] = 12,22
        ret.loc[len(ret)] = 13,23
        ret.loc[len(ret)] = 14,24
        ret.loc[len(ret)] = 15,25
        ret.loc[len(ret)] = 16,26
        ret.loc[len(ret)] = 17,27
        ret.loc[len(ret)] = 18,28
        ret.loc[len(ret)] = 19,29
        return ret
    
    @pytest.fixture
    def y_true(self):
        return pd.Series([
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        ])
    
    @pytest.mark.parametrize(("metric", "less_is_better", 'target'), [ 
                                                    ('test_accuracy', False, 1),
                                                    ('test_mse', True, 3),
                                                ])
    def test_best_fold_idx(self, cv_model, cv_splits, X, y_true, metric, less_is_better, target):
        best_fold, _, _, _ = get_operation_model(cv_model, cv_splits, X, y_true, metric, less_is_better)
        assert best_fold == target


    @pytest.mark.parametrize(("metric", "less_is_better", 'target'), [ 
                                                    ('test_accuracy', False, [12,22,13,23]),
                                                    ('test_mse', True, [16,26,17,27]),
                                                ])
    def test_best_x(self, cv_model, cv_splits, X, y_true, metric, less_is_better, target):
        _, best_x, _, _ = get_operation_model(cv_model, cv_splits, X, y_true, metric, less_is_better)
        assert best_x.shape[0] == 2
        assert best_x.shape[1] == 2
        assert best_x.x1.iloc[0] == target[0]
        assert best_x.x2.iloc[0] == target[1]
        assert best_x.x1.iloc[1] == target[2]
        assert best_x.x2.iloc[1] == target[3]


    @pytest.mark.parametrize(("metric", "less_is_better", 'target'), [ 
                                                    ('test_accuracy', False, [32,33]),
                                                    ('test_mse', True, [36,37]),
                                                ])
    def test_best_y(self, cv_model, cv_splits, X, y_true, metric, less_is_better, target):
        _, _, best_y, _ = get_operation_model(cv_model, cv_splits, X, y_true, metric, less_is_better)
        assert best_y.shape[0] == 2
        assert best_y.iloc[0] == target[0]
        assert best_y.iloc[1] == target[1]


    @pytest.mark.parametrize(("metric", "less_is_better", 'target'), [ 
                                                    ('test_accuracy', False, 'model_41'),
                                                    ('test_mse', True, 'model_43'),
                                                ])
    def test_best_model(self, cv_model, cv_splits, X, y_true, metric, less_is_better, target):
        _, _, _, model = get_operation_model(cv_model, cv_splits, X, y_true, metric, less_is_better)
        assert model == target
