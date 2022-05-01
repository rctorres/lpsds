"""Test file foor pre_process.py module"""

import pytest
import pandas as pd
from utils import keep, ObjectView, to_list, smart_tuple

        
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
