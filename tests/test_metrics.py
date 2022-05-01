"""Tests metrics module"""

import pytest
import math
import numpy as np
from metrics import sp_index

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
                                                    (np.array([0.9, 0.2, 0.4]), np.array([0.89, 0.4, 0.4])),
                                                    (np.array([ [0.9, 0.2, 0.4] ]), np.array([ [0.89, 0.4, 0.4] ])),
                                                    (np.array([ [0.9], [0.2], [0.4] ]), np.array([ [0.89], [0.4], [0.4] ])),
                                                ])
    def test_arrays(self, vec1, vec2):
        """Tests array case"""
        ret = sp_index(vec1, vec2)
        for v1, v2, r in zip(vec1.flatten(), vec2.flatten(), ret):
            assert r == self.sp_func(v1,v2)
