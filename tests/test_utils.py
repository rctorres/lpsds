"""Test file foor pre_process.py module"""

import pytest
import pandas as pd
from utils import keep

        
class TestKeep():
    @pytest.fixture
    def df(self):
        data = {}
        data['a'] = [11,22,33,44,55]
        data['b'] = [111,222,333,444,555]
        data['c'] = [True,True,False,True,False]
        data['d'] = ['v1','v2','v3','v4','v5']
        return pd.DataFrame(data)

    def test_remove_index(self, df):
        to_keep = df.a.isin([22,44])
        keep(df, index=to_keep)
        assert len(df) == 2
        assert df.index[0] == 1
        assert df.index[1] == 3

    def test_remove_column(self, df):
        to_keep = df.columns.isin(['a','d'])
        keep(df, columns=to_keep)
        assert len(df.columns) == 2
        assert df.columns[0] == 'a'
        assert df.columns[1] == 'd'
    
    def test_remove_index_and_columns(self, df):
        idx_to_keep = df.a.isin([11,22,55])
        cols_to_keep = df.columns == 'c'
        keep(df, index=idx_to_keep, columns=cols_to_keep)
        
        #Checking index
        assert len(df) == 3
        assert df.index[0] == 0
        assert df.index[1] == 1
        assert df.index[2] == 4
        
        #Checking columns
        assert len(df.columns) == 1
        assert df.columns[0] == 'c'

    def test_inplace_true(self, df):
        idx_to_keep = df.a.isin([33,44])
        ret = keep(df, index=idx_to_keep, inplace=True)
        assert ret is None

    def test_inplace_false(self, df):
        idx_to_keep = df.a.isin([33,44])
        ret = keep(df, index=idx_to_keep, inplace=False)
        
        #Ckecking that ret is not None and that df is unchanged.
        assert ret is not None
        assert len(df) == 5
        assert len(ret) == 2
