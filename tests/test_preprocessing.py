import pytest
import pandas as pd
import numpy as np
from lpsds.preprocessing import StandardNaN, BooleanEncode, SmartFloatCasting, drop_null_cols


class TestDropNullCols:
    """Test drop_null_cols"""

    @pytest.fixture
    def df(self):
        """Test dataset"""

        return pd.DataFrame({
            'a' : [1,2,3,None],
            'b' : [111,np.nan,np.nan, None],
            'c' : [11,22,np.nan, None],
            'd' : [None,np.nan,np.nan, None],
        })
    
    def test_operation(self, df):
        """Test normal behavior"""
        ret = drop_null_cols(df, threshold=0.5)
        cols = ret.columns
        assert 'a' in cols
        assert 'b' not in cols
        assert 'c' in cols
        assert 'd' not in cols


    def test_inplace_true(self, df):
        """Test inplace True"""
        drop_null_cols(df, threshold=0.5, inplace=True)
        cols = df.columns
        assert 'a' in cols
        assert 'b' not in cols
        assert 'c' in cols
        assert 'd' not in cols

    
    def test_inplace_false(self, df):
        """Test inplace False"""
        drop_null_cols(df, threshold=0.5, inplace=False)
        cols = df.columns
        assert 'a' in cols
        assert 'b' in cols
        assert 'c' in cols
        assert 'd' in cols
    
    def test_std_nan_true(self):
        """Test NaN standartization"""
        df = pd.DataFrame({
            'a' : ['1','','NA','None'],
            'b' : [111,np.nan,np.nan, None],
            'c' : [11,22,np.nan, None],
            'd' : [None,'nan','val', None],
        })
        ret = drop_null_cols(df, threshold=0.5, std_nan=True)
        cols = ret.columns
        assert 'a' not in cols
        assert 'b' not in cols
        assert 'c' in cols
        assert 'd' not in cols

    def test_std_nan_false(self):
        """Test NaN standartization"""
        df = pd.DataFrame({
            'a' : ['1','','NA','None'],
            'b' : [111,np.nan,np.nan, None],
            'c' : [11,22,np.nan, None],
            'd' : [None,'nan','val', None],
        })
        ret = drop_null_cols(df, threshold=0.5, std_nan=False)
        cols = ret.columns
        assert 'a' in cols
        assert 'b' not in cols
        assert 'c' in cols
        assert 'd' in cols



class TestBooleanEncode:
    """Test BooleanEncode class"""

    @pytest.fixture
    def bool_map(self):
        """boolean map"""
        return {
            'pos' : +1,
            'neg' : -1,
            'None' : 0,
        }
    
    @pytest.fixture
    def df(self):
        """Test dataset"""
        return pd.DataFrame({
            'a' : ['pos', 'neg', 'None'],
            'b' : ['pos', 'pos', 'neg']
        })


    def test_more_3_values(self):
        """
        ValueError must be raised 
        if the map has mopre than 3 distinct values for items
        """
        bool_map = dict(
            pos=+1,
            neg=-1,
            other=0,
            another_pos=+2,
        )
        with pytest.raises(ValueError):
            BooleanEncode(bool_map)

    def test_more_3_keys(self):
        """
        No error must occur if more than 3 keys are given as long as
        values are no more than 3
        """
        bool_map = dict(
            pos=+1,
            neg=-1,
            other=0,
            another_pos=+1,
        )
        BooleanEncode(bool_map)
        assert True

    def test_normal_operation(self, bool_map, df):
        """Test normal operation"""
        obj = BooleanEncode(bool_map)
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == +1
        assert ret.a.iloc[1] == -1
        assert ret.a.iloc[2] == 0

        assert ret.b.iloc[0] == +1
        assert ret.b.iloc[1] == +1
        assert ret.b.iloc[2] == -1


    def test_normal_operation_boolean(self):
        """Test normal operation"""
        test_map = {
            True : +1,
            False : -1,
            np.nan : 0,
        }

        df = pd.DataFrame({
            'a' : [True, False, np.nan],
            'b' : [True, True, False]
        })

        obj = BooleanEncode(test_map)
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == +1
        assert ret.a.iloc[1] == -1
        assert ret.a.iloc[2] == 0

        assert ret.b.iloc[0] == +1
        assert ret.b.iloc[1] == +1
        assert ret.b.iloc[2] == -1


    def test_inplace_true(self, bool_map, df):
        """Test inplace True"""
        obj = BooleanEncode(bool_map, inplace=True)
        obj.fit_transform(df)
        assert df.a.iloc[0] == +1
        assert df.a.iloc[1] == -1
        assert df.a.iloc[2] == 0

        assert df.b.iloc[0] == +1
        assert df.b.iloc[1] == +1
        assert df.b.iloc[2] == -1

    def test_inplace_false(self, bool_map, df):
        """Test inplace False"""
        obj = BooleanEncode(bool_map, inplace=False)
        obj.fit_transform(df)
        assert df.a.iloc[0] == 'pos'
        assert df.a.iloc[1] == 'neg'
        assert df.a.iloc[2] == 'None'

        assert df.b.iloc[0] == 'pos'
        assert df.b.iloc[1] == 'pos'
        assert df.b.iloc[2] == 'neg'

    def test_dtype(self, bool_map, df):
        """Test dtype override"""
        obj = BooleanEncode(bool_map, dtype=np.int8)
        ret = obj.fit_transform(df)
        assert str(ret.a.dtype) == 'int8'
        assert str(ret.b.dtype) == 'int8'





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



class TestSmartFloatCasting:
    """Test SmartFloatCasting"""
    
    @pytest.fixture
    def df(self):
        """Test dataset"""
        return pd.DataFrame({
            'a' : ['1,23', '-5',],
            'b' : ['-23.53', '0',],
        })

    def test_normal_operation(self, df):
        """test normal operation"""
        obj = SmartFloatCasting()
        ret = obj.fit_transform(df)
        assert ret.a.iloc[0] == 1.23
        assert ret.a.iloc[1] == -5
        assert ret.b.iloc[0] == -23.53
        assert ret.b.iloc[1] == 0

    def test_inplace_true(self, df):
        """test inplace True"""
        obj = SmartFloatCasting(inplace=True)
        obj.fit_transform(df)
        assert df.a.iloc[0] == 1.23
        assert df.a.iloc[1] == -5
        assert df.b.iloc[0] == -23.53
        assert df.b.iloc[1] == 0

    def test_inplace_false(self, df):
        """test inplace False"""
        obj = SmartFloatCasting(inplace=False)
        obj.fit_transform(df)
        assert df.a.iloc[0] == '1,23'
        assert df.a.iloc[1] == '-5'
        assert df.b.iloc[0] == '-23.53'
        assert df.b.iloc[1] == '0'

    def test_dtype(self, df):
        """test different dtype"""
        obj = SmartFloatCasting(dtype=np.float32)
        ret = obj.fit_transform(df)
        assert str(ret.a.dtype) == 'float32'
        assert str(ret.b.dtype) == 'float32'
