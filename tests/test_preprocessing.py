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

    def test_simple_case(self, obj):
        """Test simple behavior"""

        df = pd.DataFrame({
            'a' : ['aaa', 'None', 'NA'], 
            'b' : ['', 'bbb', 'NAN'],
        })
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == 'aaa'
        assert np.isnan(ret.a.iloc[1])
        assert np.isnan(ret.a.iloc[2])

        assert np.isnan(ret.b.iloc[0])
        assert ret.b.iloc[1] == 'bbb'
        assert np.isnan(ret.b.iloc[2])


    def test_nan_rep_addition(self):
        """Test adding new nan rep"""

        df = pd.DataFrame({
            'a' : ['aaa', 'None', 'ccc'], 
            'b' : ['', 'bbb', 'NAN'],
        })

        obj = StandardNaN(additional_nan_rep=['bbb', 'ccc'])
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == 'aaa'
        assert np.isnan(ret.a.iloc[1])
        assert np.isnan(ret.a.iloc[2])

        assert np.isnan(ret.b.iloc[0])
        assert np.isnan(ret.b.iloc[1])
        assert np.isnan(ret.b.iloc[2])


    def test_std_nan_override(self):
        """Testing setting a new NaN override"""

        df = pd.DataFrame({
            'a' : ['aaa', 'None', 'ccc'], 
            'b' : ['', 'bbb', 'NAN'],
        })

        std_val = 'NEW'
        obj = StandardNaN(additional_nan_rep=['aaa'], std_nan_val=std_val)
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == std_val
        assert ret.a.iloc[1] == std_val
        assert ret.a.iloc[2] == 'ccc'

        assert ret.b.iloc[0] == std_val
        assert ret.b.iloc[1] == 'bbb'
        assert ret.b.iloc[2] == std_val


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


    def test_numeric_columns(self, obj):
        """Test if the class can handle dataframe with numeric columns"""

        df = pd.DataFrame({
            'a' : [1, 2, np.nan], 
            'b' : ['nan', 'To Keep', '<na>'],
            'c' : [4.1, np.nan, 4.3],
            'd' : ['To Keep', '<na>', '<Na>'],
        })
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == 1
        assert ret.a.iloc[1] == 2
        assert np.isnan(ret.a.iloc[2])

        assert np.isnan(ret.b.iloc[0])
        assert ret.b.iloc[1] == 'To Keep'
        assert np.isnan(ret.b.iloc[2])

        assert ret.c.iloc[0] == 4.1
        assert np.isnan(ret.c.iloc[1])
        assert ret.c.iloc[2] == 4.3

        assert ret.d.iloc[0] == 'To Keep'
        assert np.isnan(ret.d.iloc[1])
        assert np.isnan(ret.d.iloc[2])

    def test_inplace_true(self):
        """Test inplace = True"""

        df = pd.DataFrame({
            'a' : ['To Keep', '<na>'], 
            'b' : ['None', 'To Keep'], 
        })

        obj = StandardNaN(inplace=True)
        obj.fit_transform(df)

        assert df.a.iloc[0] == 'To Keep'
        assert np.isnan(df.a.iloc[1])
        assert np.isnan(df.b.iloc[0])
        assert df.b.iloc[1] == 'To Keep'


    def test_inplace_false(self):
        """Test inplace = False"""

        df = pd.DataFrame({
            'a' : ['To Keep', '<na>'], 
            'b' : ['None', 'To Keep'], 
        })

        obj = StandardNaN(inplace=False)
        obj.fit_transform(df)

        assert df.a.iloc[0] == 'To Keep'
        assert df.a.iloc[1] == '<na>'
        assert df.b.iloc[0] == 'None'
        assert df.b.iloc[1] == 'To Keep'
