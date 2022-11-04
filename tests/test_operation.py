import pytest
import numpy as np
import pandas as pd
from lpsds.operation import get_operation_model, get_fold_data


class FoldDataBase:
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


class TestGetFoldData(FoldDataBase):

    def test_x_trn(self, cv_model, cv_splits, X, y_true):
        x_trn, _, _, _, _ = get_fold_data(cv_model, cv_splits, X, y_true, 1)
        assert x_trn.shape[0] == 8
        assert x_trn.shape[1] == 2
        assert x_trn.x1.iloc[0] == 10
        assert x_trn.x2.iloc[0] == 20
        assert x_trn.x1.iloc[1] == 11
        assert x_trn.x2.iloc[1] == 21
        assert x_trn.x1.iloc[2] == 14
        assert x_trn.x2.iloc[2] == 24
        assert x_trn.x1.iloc[3] == 15
        assert x_trn.x2.iloc[3] == 25
        assert x_trn.x1.iloc[4] == 16
        assert x_trn.x2.iloc[4] == 26
        assert x_trn.x1.iloc[5] == 17
        assert x_trn.x2.iloc[5] == 27
        assert x_trn.x1.iloc[6] == 18
        assert x_trn.x2.iloc[6] == 28
        assert x_trn.x1.iloc[7] == 19
        assert x_trn.x2.iloc[7] == 29


    def test_x_tst(self, cv_model, cv_splits, X, y_true):
        _, x_tst, _, _, _ = get_fold_data(cv_model, cv_splits, X, y_true, 1)
        assert x_tst.shape[0] == 2
        assert x_tst.shape[1] == 2
        assert x_tst.x1.iloc[0] == 12
        assert x_tst.x2.iloc[0] == 22
        assert x_tst.x1.iloc[1] == 13
        assert x_tst.x2.iloc[1] == 23


    def test_y_trn(self, cv_model, cv_splits, X, y_true):
        _, _, y_trn, _, _ = get_fold_data(cv_model, cv_splits, X, y_true, 1)
        assert y_trn.shape[0] == 8
        assert y_trn.iloc[0] == 30
        assert y_trn.iloc[1] == 31
        assert y_trn.iloc[2] == 34
        assert y_trn.iloc[3] == 35
        assert y_trn.iloc[4] == 36
        assert y_trn.iloc[5] == 37
        assert y_trn.iloc[6] == 38
        assert y_trn.iloc[7] == 39


    def test_y_tst(self, cv_model, cv_splits, X, y_true):
        _, _, _, y_tst, _ = get_fold_data(cv_model, cv_splits, X, y_true, 1)
        assert y_tst.shape[0] == 2
        assert y_tst.iloc[0] == 32
        assert y_tst.iloc[1] == 33


    def test_model(self, cv_model, cv_splits, X, y_true):
        _, _, _, _, model = get_fold_data(cv_model, cv_splits, X, y_true, 1)
        assert model == 'model_41'


    def test_numpy_data_x_trn(self, cv_model, cv_splits, X, y_true):
        x_trn, _, _, _, _ = get_fold_data(cv_model, cv_splits, X.to_numpy(), y_true.to_numpy(), 2)
        assert x_trn.shape[0] == 8
        assert x_trn.shape[1] == 2
        assert x_trn[0,0] == 10
        assert x_trn[0,1] == 20
        assert x_trn[1,0] == 11
        assert x_trn[1,1] == 21
        assert x_trn[2,0] == 12
        assert x_trn[2,1] == 22
        assert x_trn[3,0] == 13
        assert x_trn[3,1] == 23
        assert x_trn[4,0] == 16
        assert x_trn[4,1] == 26
        assert x_trn[5,0] == 17
        assert x_trn[5,1] == 27
        assert x_trn[6,0] == 18
        assert x_trn[6,1] == 28
        assert x_trn[7,0] == 19
        assert x_trn[7,1] == 29


    def test_numpy_data_x_test(self, cv_model, cv_splits, X, y_true):
        _, x_tst, _, _, _ = get_fold_data(cv_model, cv_splits, X.to_numpy(), y_true.to_numpy(), 2)
        assert x_tst.shape[0] == 2
        assert x_tst.shape[1] == 2
        assert x_tst[0,0] == 14
        assert x_tst[0,1] == 24
        assert x_tst[1,0] == 15
        assert x_tst[1,1] == 25



    def test_numpy_data_y_trn(self, cv_model, cv_splits, X, y_true):
        _, _, y_trn, _, _ = get_fold_data(cv_model, cv_splits, X.to_numpy(), y_true.to_numpy(), 2)
        assert y_trn.shape[0] == 8
        assert y_trn[0] == 30
        assert y_trn[1] == 31
        assert y_trn[2] == 32
        assert y_trn[3] == 33
        assert y_trn[4] == 36
        assert y_trn[5] == 37
        assert y_trn[6] == 38
        assert y_trn[7] == 39


    def test_numpy_data_y_test(self, cv_model, cv_splits, X, y_true):
        _, _, _, y_tst, _ = get_fold_data(cv_model, cv_splits, X.to_numpy(), y_true.to_numpy(), 2)
        assert y_tst.shape[0] == 2
        assert y_tst[0] == 34
        assert y_tst[1] == 35



class TestGetOperationModel(FoldDataBase):

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


    def test_numpy_data_x(self, cv_model, cv_splits, X, y_true):
        _, best_x, _, _ = get_operation_model(cv_model, cv_splits, X.to_numpy(), y_true, 'test_accuracy')
        assert best_x.shape[0] == 2
        assert best_x.shape[1] == 2
        assert best_x[0,0] == 12
        assert best_x[0,1] == 22
        assert best_x[1,0] == 13
        assert best_x[1,1] == 23


    @pytest.mark.parametrize(("y"), [
                                        (pd.Series([30,31,32,33,34,35,36,37,38,39]).to_numpy()),
                                        (np.array([30,31,32,33,34,35,36,37,38,39])),
                                        (np.array([30,31,32,33,34,35,36,37,38,39]).reshape(-1,1)),
                                    ])
    def test_numpy_data_y(self, cv_model, cv_splits, X, y):
        _, _, best_y, _ = get_operation_model(cv_model, cv_splits, X, y, 'test_accuracy')
        assert best_y.shape[0] == 2
        assert best_y[0] == 32
        assert best_y[1] == 33
