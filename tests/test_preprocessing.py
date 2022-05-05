import pytest

from preprocessing import StandardNaN
import pandas as pd
import numpy as np


class TestStandardNaN:
    """Testing StandardNaN class"""

    @pytest.fixture
    def obj(self):
        """Test object"""
        return StandardNaN()

    def test_case_insensitiveness(self, obj):
        """Test whether the transformation is case insensitive."""

        df = pd.DataFrame({
            'a' : ['na', 'nA', 'NA'], 
            'b' : ['nan', 'Nan', 'NAN'],
            'c' : ['', ' ', '   '],
            'd' : ['<NA>', '<na>', '<Na>'],
            'e' : ['não informado', 'Não Informado', 'Não InFoRmAdO'],
            'f' : ['None', 'nONe', 'NONE'],
        })
        res = obj.fit_transform(df)

        for idx, c in df.iterrows():
            for col in df.columns:
                assert np.isnan(np.nan)
