"""Test file foor pre_process.py module"""

import pytest
import numpy as np
import pandas as pd
from lpsds.utils import keep, ObjectView, to_list, smart_tuple, confusion_matrix_annotation

        
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


class TestObjectView:
    """Test ObjectView class"""

    @pytest.fixture
    def obj(self):
        """Default ObjectView object"""
        return ObjectView()

    def test_set_value(self, obj):
        """Test if the obj.var =  value works"""
        obj.my_field = 10
        assert obj['my_field'] == 10

    def test_get_value(self, obj):
        """Test if we can read the value using the notation 'obj.var'"""
        obj['my_field'] = 10
        assert obj.my_field == 10

    def test_attribute_not_found(self, obj):
        """Test if an AttributeError is raised if a required field does not exists."""
        obj.my_field = 10
        with pytest.raises(AttributeError):
            obj.my_other_field
    
    def test_attribute_not_found_for_deletion(self, obj):
        """Test if an AttributeError is raised if a required field does not exists upon deletion."""
        obj.my_field = 10
        with pytest.raises(AttributeError):
            del obj.my_other_field



class TestToList:
    """Test to_list function"""

    def test_value_to_list(self):
        """Test if a value gets converted to a list"""
        ret = to_list(10)
        assert type(ret) is list
        assert len(ret) == 1
        assert ret[0] == 10

    def assert_list_remains_as_list(self):
        """A passed list must be returned unchanged"""
        val = [1]
        ret = to_list(val)
        assert type(ret) is list
        assert len(ret) == 1
        assert ret[0] == 10

    def assert_list_remains_as_tuple(self):
        """A passed tuple must be returned unchanged"""
        val = (10, 20)
        ret = to_list(val)
        assert type(ret) is tuple
        assert len(ret) == 2
        assert ret[0] == 10
        assert ret[1] == 20

    def assert_string_remains_as_string(self):
        """A passed string must be returned unchanged"""
        val = "my test string"
        ret = to_list(val)
        assert type(ret) is str
        assert ret == val


class TestSmartTuple:
    """Test smart_tuple function"""

    def test_single_value_tuple(self):
        val = (10,)
        assert smart_tuple(val) == 10

    def test_multi_value_tuple(self):
        val = (10, 20, 30)
        ret = smart_tuple(val)
        assert type(ret) is tuple
        assert len(ret) == 3
        assert ret[0] == 10
        assert ret[1] == 20
        assert ret[2] == 30



class TestCMAnnotate:

    @pytest.fixture
    def cm(self):
        return np.array([
            [[1,7,20],[40,5,16]],
            [[19,1,9], [10,21,2]],
            [[17,14,15],[26,17,18]],
            [[19,40,21], [32,23,24]]
        ])
    
    @pytest.fixture
    def fmt_str(self):
        return '${mean:.2f}^{{+{e_max:.2f}}}_{{-{e_min:.2f}}}$'
    
    @pytest.fixture
    def seed(self):
        return 23
    
    
    def test_ret_shape(self, cm):
        ret = confusion_matrix_annotation(cm)
        assert len(ret.shape) == 2
        assert ret.shape[0] == 2
        assert ret.shape[1] == 3
    

    def test_difference_false(self, cm, fmt_str, seed):
        ret = confusion_matrix_annotation(cm, use_difference=False, seed=seed)
        assert ret.iloc[0][0] == fmt_str.format(mean=14.00, e_min=5.49, e_max=19.00)
        assert ret.iloc[0][1] == fmt_str.format(mean=15.50, e_min=4.00, e_max=31.75)
        assert ret.iloc[0][2] == fmt_str.format(mean=16.25, e_min=11.75, e_max=20.50)

        assert ret.iloc[1][0] == fmt_str.format(mean=27.00, e_min=15.50, e_max=36.50)
        assert ret.iloc[1][1] == fmt_str.format(mean=16.50, e_min=8.98, e_max=22.00)
        assert ret.iloc[1][2] == fmt_str.format(mean=15.00, e_min=6.00, e_max=22.00)


    def test_difference_true(self, cm, fmt_str, seed):
        ret = confusion_matrix_annotation(cm, use_difference=True, seed=seed)
        assert ret.iloc[0][0] == fmt_str.format(mean=14.00, e_min=8.51, e_max=5.00)
        assert ret.iloc[0][1] == fmt_str.format(mean=15.50, e_min=11.50, e_max=16.25)
        assert ret.iloc[0][2] == fmt_str.format(mean=16.25, e_min=4.5, e_max=4.25)

        assert ret.iloc[1][0] == fmt_str.format(mean=27.00, e_min=11.50, e_max=9.50)
        assert ret.iloc[1][1] == fmt_str.format(mean=16.50, e_min=7.52, e_max=5.50)
        assert ret.iloc[1][2] == fmt_str.format(mean=15.00, e_min=9.00, e_max=7.00)


    def test_fmt_str(self, cm, seed):
        fmt_str = '{mean:.0f}, {e_min:.0f}, {e_max:.0f}'
        ret = confusion_matrix_annotation(cm, use_difference=True, seed=seed, fmt_str=fmt_str)
        assert ret.iloc[0][0] == fmt_str.format(mean=14, e_min=9, e_max=5)
        assert ret.iloc[0][1] == fmt_str.format(mean=16, e_min=12, e_max=16)
        assert ret.iloc[0][2] == fmt_str.format(mean=16, e_min=4, e_max=4)

        assert ret.iloc[1][0] == fmt_str.format(mean=27, e_min=12, e_max=10)
        assert ret.iloc[1][1] == fmt_str.format(mean=16, e_min=8, e_max=6)
        assert ret.iloc[1][2] == fmt_str.format(mean=15, e_min=9, e_max=7)
