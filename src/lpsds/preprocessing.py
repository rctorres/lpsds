"""
Handles data pre_processing related operations.
"""

import re
import numpy as np


def cnpj_format(cnpj, ignore_null=False):
    """
    cnpj_format(cnpj, ignore_null=False)

    Returns a standartized (xx.xxx.xxx/xxxx-xx) version of the provided CNPJ.
    If the supplied CNPJ has less than 14 digits, this function will enforce
    it to have 14 digits by padding zeros to the left side of the CNPJ.
    
    If ignore_null is False (default), Null values will be converted to 00.000.000/0000-00.
    Otherwise, it will be returned as is (no change to it).
    """
    
    if ignore_null and cnpj in [None, np.nan]: return cnpj
    
    #Making sur that all CNPJ has only digits.
    non_digits = re.compile(r'\D')
    cnpj = non_digits.sub('', str(cnpj))

    #If the digits-only CNPJ has less than 14 digits, we complete it by
    #adding zeros to the left..
    digitos_faltando = 14 - len(cnpj)
    cnpj = '0'*digitos_faltando + cnpj

    #Formating it as "xx.xxx.xxx/xxxx-xx".
    filt = re.compile(r'(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})')
    g = filt.match(cnpj)
    return f'{g.group(1)}.{g.group(2)}.{g.group(3)}/{g.group(4)}-{g.group(5)}'


def standard_nan(df, additional_nan_rep={}, std_nan_val=np.NaN, inplace=False):
    """
    Standartized multiple values that should be considered as null into a single value.
    """
    to_replace = {
        'na' : std_nan_val,
        'nan' : std_nan_val,
        '' : std_nan_val,
        '<NA>' : std_nan_val,
        'NÃO INFORMADO' : std_nan_val,
    }
    
    to_replace.update(additional_nan_rep)
    return df.replace(to_replace, inplace=inplace)


def drop_null_cols(df, threshold=0.6, inplace=False):
    """
    Remove columns which number of null samples is greater than the specified threshold.
    """
    null_freq = df.isna().sum() / len(df)
    df = df.drop(columns=null_freq.loc[null_freq > threshold].index, inplace=inplace)
    return df



def get_itb_range_averages(cat_val, is_integer=False):
    """
    Returns the mean value of ITB ranges for revenue, employees number, etc.
    """
    
    def get_number(val, is_integer=False):
        if not val: return np.NaN
        filt = re.compile(r'[\.,]')
        if is_integer: val = filt.sub('', val)
        return float(val)

    def get_order(val):
        orders_map = {
            'K' : 10**3,
            'M' : 10**6,
            'B' : 10**9,
        }
        return 1 if not val else orders_map[val[0]]

    filt = re.compile(r'(R\$)?\s*([\d\.,]+)\s*([K,M,I,B]*)\s*\-\s*(R\$)?\s*([\d\.,]+)\s*([K,M,I,B]*)\s*|\+\s*(R\$)?\s*([\d\.,]+)\s*\s*([K,M,I,B]*)')
    a = filt.match(cat_val)
    vec = np.array([
        get_number(a.group(2), is_integer) * get_order(a.group(3)),
        get_number(a.group(5), is_integer) * get_order(a.group(6)),
        get_number(a.group(8), is_integer) * get_order(a.group(9)),
        ])

    return np.nanmean(vec)


class RangeAverage:
    def __init__(self, is_integer, dtype=np.float64, inplace=True):
        self.is_integer = is_integer
        self.dtype = dtype
        self.inplace = inplace
    
    def fit(self, X, y=None, **kwargs):
        return None
    
    def transform(self, X, **kwargs):
        if not self.inplace: X = X.copy()
        for c in X.columns:
            X[c] = X[c].apply(get_itb_range_averages, is_integer=self.is_integer).astype(self.dtype)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)

    
class BooleanEncode:
    def __init__(self, boolean_map, inplace=True, dtype=None):
        self.boolean_map = boolean_map
        self.inplace = inplace
        self.dtype = dtype
    
    def fit(self, X, y=None, **kwargs):
        return None
    
    def transform(self, X, **kwargs):
        if not self.inplace: X = X.copy()
        for c in X.columns:
            X[c].replace(self.boolean_map, inplace=True)
            if self.dtype is not None: X[c] = X[c].astype(self.dtype)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)


    
class StandardNaN:
    """
    Standartizes multiple NaN representations into an unique representation.

    Ex: "", "Not Informed", "Not Available", etc -> np.NaN
    """

    def __init__(self, additional_nan_rep: list=[], std_nan_val=np.NaN, inplace: bool=True):
        """
        def __init__(self, additional_nan_rep: dict={}, std_nan_val=np.NaN, inplace: bool=True):
        Class constructor

        Input parameters:
          - additional_nan_rep: a list with *additional* values that you wish to standartize
            with an unique NaN representation. By default, the following values are already
            included: 'na', 'nan', '', '<NA>', 'NÃO INFORMADO'.
          - std_nan_val: the unique NaN representation that will be used instead of the 
            representations passed.
          - inplace: whether to work on a copy of the data or not.
        """

        #Original NaN reps
        self.nan_map = {
            'na' : std_nan_val,
            'nan' : std_nan_val,
            '' : std_nan_val,
            '<NA>' : std_nan_val,
            'NÃO INFORMADO' : std_nan_val,
        }

        #Adding any custom value brought by the user.
        for v in additional_nan_rep:
            self.nan_map[v] = std_nan_val

        self.inplace = inplace
    
    def fit(self, X, y=None, **kwargs):
        return None
    
    def transform(self, X, **kwargs):
        if not self.inplace: X = X.copy()
        return X.replace(self.nan_map)
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)


class SmartFloatCasting:
    def __init__(self, orig_dec_sep=',', dest_dec_sep='.', inplace=True):
        self.orig_dec_sep = orig_dec_sep
        self.dest_dec_sep = dest_dec_sep
        self.inplace = inplace
    
    def fit(self, X, y=None, **kwargs):
        return None

    def transform(self, X, **kwargs):
        if not self.inplace: X = X.copy()
        for c in X.columns:
            X[c] = X[c].str.replace(self.orig_dec_sep, self.dest_dec_sep).astype(np.float64)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        return self.transform(X)
