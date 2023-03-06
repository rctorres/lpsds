"""Test file foor pre_process.py module"""

import pytest
import numpy as np
import pandas as pd
import sklearn.pipeline
from lpsds.utils import keep, ObjectView, to_list, smart_tuple, confusion_matrix_annotation, pretty_title, error_bar_roc, pipeline_split

        
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


class TestPrettyTitle:

    @pytest.mark.parametrize(("val", "ret"), [ 
                                                ('auc', 'AUC'),
                                                ('Auc', 'AUC'),
                                                ('aUc', 'AUC'),
                                                ('auC', 'AUC'),
                                                ('AUc', 'AUC'),
                                                ('aUC', 'AUC'),
                                                ('AUC', 'AUC'),
                                                ('specificity', 'Specificity'),
                                                ('Specificity', 'Specificity'),
                                                ('SPECIFICITY', 'Specificity'),
                                            ])
    def test_normal_case(self, val, ret):
        assert pretty_title(val) == ret


    @pytest.mark.parametrize(("val", "ret"), [ 
                                                ('auc', 'Auc'),
                                                ('Auc', 'Auc'),
                                                ('aUc', 'Auc'),
                                                ('auC', 'Auc'),
                                                ('AUc', 'Auc'),
                                                ('aUC', 'Auc'),
                                                ('AUC', 'Auc'),
                                                ('specificity', 'Specificity'),
                                                ('Specificity', 'Specificity'),
                                                ('SPECIFICITY', 'Specificity'),
                                            ])
    def test_new_threshold(self, val, ret):
        assert pretty_title(val, abbrev_threshold=2) == ret


    @pytest.mark.parametrize(("val", "ret"), [ 
                                                ('auc', 'AUC'),
                                                ('Auc', 'AUC'),
                                                ('aUc', 'AUC'),
                                                ('auC', 'AUC'),
                                                ('AUc', 'AUC'),
                                                ('aUC', 'AUC'),
                                                ('AUC', 'AUC'),
                                                ('specificity', 'SPECIFICITY'),
                                                ('Specificity', 'SPECIFICITY'),
                                                ('SPECIFICITY', 'SPECIFICITY'),
                                            ])
    def test_new_threshold_large(self, val, ret):
        assert pretty_title(val, abbrev_threshold=20) == ret




class TestErrorBarRoc:

    @pytest.fixture
    def roc_df(self):
        df = pd.DataFrame(columns=['fold', 'false_positive', 'true_positive', 'threshold'])
        df.loc[len(df)] = 0, 0, 0, 0 # tp = 2*fp
        df.loc[len(df)] = 0, 0.2, 0.4, 0
        df.loc[len(df)] = 0, 0.3, 0.6, 0
        df.loc[len(df)] = 0, 1.0, 2.0, 0
        df.loc[len(df)] = 1, 0, 0, 0 # tp = 10*fp
        df.loc[len(df)] = 1, 0.6, 6, 0
        df.loc[len(df)] = 1, 1, 10, 0
        return df

    @pytest.fixture
    def ret(self, roc_df):
        return error_bar_roc(roc_df, num_points=5)

    def test_ret_size(self, ret):
        assert len(ret) == 10


    def test_col_names(self, ret):
        assert len(ret.columns) == 3
        assert ret.columns[0] == 'false_positive'
        assert ret.columns[1] == 'true_positive'
        assert ret.columns[2] == 'fold'
    
    def test_fold_0(self, ret):
        fold = ret.loc[ret.fold==0]
        assert len(fold) == 5
        assert fold.iloc[0].true_positive == 0
        assert fold.iloc[1].true_positive == 0.5
        assert fold.iloc[2].true_positive == 1
        assert fold.iloc[3].true_positive == 1.5
        assert fold.iloc[4].true_positive == 2


    def test_fold_1(self, ret):
        fold = ret.loc[ret.fold==1]
        assert len(fold) == 5
        assert fold.iloc[0].true_positive == 0
        assert fold.iloc[1].true_positive == 2.5
        assert fold.iloc[2].true_positive == 5.0
        assert fold.iloc[3].true_positive == 7.5
        assert fold.iloc[4].true_positive == 10

    def test_fp_(self, ret):
        fold0 = ret.loc[ret.fold==0]
        assert fold0.iloc[0].false_positive == 0
        assert fold0.iloc[1].false_positive == 0.25
        assert fold0.iloc[2].false_positive == 0.5
        assert fold0.iloc[3].false_positive == 0.75
        assert fold0.iloc[4].false_positive == 1.0


    def test_fp_equal(self, ret):
        fold0 = ret.loc[ret.fold==0]
        fold1 = ret.loc[ret.fold==1]
        assert fold0.iloc[0].false_positive == fold1.iloc[0].false_positive
        assert fold0.iloc[1].false_positive == fold1.iloc[1].false_positive
        assert fold0.iloc[2].false_positive == fold1.iloc[2].false_positive
        assert fold0.iloc[3].false_positive == fold1.iloc[3].false_positive
        assert fold0.iloc[4].false_positive == fold1.iloc[4].false_positive
    


class TestPipelineSplit:

    class PreProc1:
        def transform(self, X):
            return X + 1
    class PreProc2:
        def transform(self, X):
            return X + 10
    class Estimator:
        def predict(self, X):
            return np.array([11,22])


    @pytest.fixture
    def pipe(self):
        return sklearn.pipeline.Pipeline(steps = [
            ('pp1', TestPipelineSplit.PreProc1()),
            ('pp2', TestPipelineSplit.PreProc2()),
            ('est', TestPipelineSplit.Estimator()),
        ])
    
    def test_pre_proc_split(self, pipe):
        pp, mod, x = pipeline_split(pipe)
        assert len(pp) == 2
        assert pp[0][0] == 'pp1'
        assert isinstance(pp[0][1], TestPipelineSplit.PreProc1)
        assert pp[1][0] == 'pp2'
        assert isinstance(pp[1][1], TestPipelineSplit.PreProc2)


    def test_estimator(self, pipe):
        pp, mod, x = pipeline_split(pipe)
        assert mod[0] == 'est'
        assert isinstance(mod[1], TestPipelineSplit.Estimator)


    def test_x_none(self, pipe):
        pp, mod, x = pipeline_split(pipe)
        assert x is None


    def test_x(self, pipe):
        X = np.array([[1,2],
                      [3,4]])
        pp, mod, x = pipeline_split(pipe, X)
        assert X.shape[0] == 2
        assert X.shape[1] == 2
        assert x[0][0] == 12
        assert x[0][1] == 13
        assert x[1][0] == 14
        assert x[1][1] == 15
