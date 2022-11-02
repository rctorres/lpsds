import pytest
import numpy as np
import pandas as pd


class TestGetOperationModel:

    @pytest.fixture
    def cv_model(self):
        return dict(
            test_accuracy=np.array([3,9,6,8,4]), #max ids = 1
            test_mse=np.array([3,9,6,2,4]), #min idx = 3
            estimator=['model_0', 'model_1', 'model_2', 'model_3', 'model_4']
        )
    
    def cv_splits(self):
        return [
            ([2,3,4,5,6,7,8,9], [0,1]), #fold 0
            ([0,1,4,5,6,7,8,9], [2,3]), #fold 1
            ([0,1,2,3,6,7,8,9], [4,5]), #fold 2
            ([0,1,2,3,4,5,8,9], [6,7]), #fold 3
            ([0,1,2,3,4,5,6,7], [8,9]), #fold 4
        ]
    
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
    
    def y_true(self):
        return ps.Series([
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
    
    