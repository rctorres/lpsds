"""Tests metrics module"""

import pytest
import math
import numpy as np
from lpsds.metrics import sp_index, sensitivity, specificity, sp_score, ppv, npv

class TestSPIndex:
    """TEsts sp_index function"""

    def sp_func(self, v1,v2):
        """Help function for SP comparison"""
        geo_mean = math.sqrt(v1*v2)
        arit_mean = 0.5 * (v1+v2)
        return math.sqrt(geo_mean * arit_mean)

    def test_scalar(self):
        """Test if scalars are passed"""
        v1,v2 = 0.9, 0.4
        assert sp_index(v1,v2) == self.sp_func(v1,v2)


    @pytest.mark.parametrize(("vec1", "vec2"), [ 
                                                    (np.array([0.9, 0.2, 0.4]), np.array([0.89, 0.4, 0.4])), # 2 dimensionless vectors
                                                    (np.array([ [0.9, 0.2, 0.4] ]), np.array([ [0.89, 0.4, 0.4] ])), #2 row vectors
                                                    (np.array([ [0.9], [0.2], [0.4] ]), np.array([ [0.89], [0.4], [0.4] ])), #2 col vectors
                                                    (np.array([0.9, 0.2, 0.4]), np.array([ [0.89, 0.4, 0.4] ])), # mixed dimensionless $ row
                                                    (np.array([0.9, 0.2, 0.4]), np.array([ [0.89], [0.4], [0.4] ])), # 2 mixed dimwnsionless & cols
                                                    (np.array([ [0.9, 0.2, 0.4] ]), np.array([0.89, 0.4, 0.4])), # mixed row & dimensionless
                                                    (np.array([ [0.9, 0.2, 0.4] ]), np.array([ [0.89], [0.4], [0.4] ])), # mixed rown & col
                                                    (np.array([ [0.9], [0.2], [0.4] ]), np.array([0.89, 0.4, 0.4])), # mixed col & dimensionless
                                                    (np.array([ [0.9], [0.2], [0.4] ]), np.array([ [0.89, 0.4, 0.4] ])), # mixed col & row

                                                ])
    def test_arrays(self, vec1, vec2):
        """Tests array case"""
        ret = sp_index(vec1, vec2)
        for v1, v2, r in zip(vec1.flatten(), vec2.flatten(), ret):
            assert r == self.sp_func(v1,v2)



class TestSKLearnMetrics:
    """Tests sensitivity"""

    @pytest.fixture
    def y_true(self):
        return np.array([1,1,1,1,1,0,0,0,0,0])

    @pytest.fixture
    def y_pred(self):
        return np.array([0,0,1,1,1,0,1,1,1,1])

    @pytest.mark.parametrize(("metric_func", "target_val"), [
        (sensitivity, 0.6),
        (specificity, 0.2),
        (sp_score, 0.37224194364083985),
        (ppv, 0.42857142857142855),
        (npv, 0.3333333333333333),
    ])
    def test_array(self, y_true, y_pred, metric_func, target_val):
        assert metric_func(y_true, y_pred) == pytest.approx(target_val, 0.0000001)

    @pytest.mark.parametrize(("metric_func"), [
        (sensitivity),
        (specificity),
        (sp_score),
        (ppv),
        (npv),
    ])
    def test_return_float(self, y_true, y_pred, metric_func):
        """Tests whether the returned value is in float format"""
        assert type(metric_func(y_true, y_pred)) == float