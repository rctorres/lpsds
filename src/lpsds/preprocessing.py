"""
Handles data pre_processing related operations.
"""

import re
import numpy as np


def drop_null_cols(df, threshold=0.6, inplace=False, std_nan=False, additional_nan_rep: list=None):
    """
    def drop_null_cols(df, threshold=0.6, inplace=False, std_nan=False, **kwargs)

    Remove columns which number of null samples is greater than the specified threshold.

    If std_nan is True, than the function will first standartise all nan values by using
    StandardNaN class. additional_nan_rep will be passed to the StandardNanClass constructor.
    """
    if not inplace: df = df.copy()
    if std_nan: StandardNaN(inplace=True, additional_nan_rep=additional_nan_rep).fit_transform(df)
    null_freq = df.isna().sum() / len(df)
    df.drop(columns=null_freq.loc[null_freq > threshold].index, inplace=True)
    return df

    
class BooleanEncode:
    """
    Creates an Boolean type encoding (+1, -1, 0) for categories that can be treated as such.
    Example: employed = ["Yes", "No", "Not Informed"] would become [+1, -1, 0].
    """
    def __init__(self, boolean_map: map, inplace: bool=True, dtype=None):
        """
        def __init__(self, boolean_map, inplace=True, dtype=None):

        input parameters:
          - boolean_map: a map where keys are categorical values and value is the number (+1, -1, 0)
            to be assigned to it.
        - inplace: whether to work on a copy of the data or not.
          - dtype: data type to cast results into.
        """

        if len(set(boolean_map.values())) > 3:
            raise ValueError('You cannot have more than 3 values to represent True, False and None')

        self.boolean_map = boolean_map
        self.inplace = inplace
        self.dtype = dtype
    
    def fit(self, X, y=None, **kwargs):
        """Dummy function. Nothing is done here."""
        return None
    
    def transform(self, X, **kwargs):
        """
        Apply the boolean encoding.
        """
        if not self.inplace: X = X.copy()
        X.replace(self.boolean_map, inplace=True)
        if self.dtype is not None: X = X.astype(self.dtype)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        """Fit + transform"""
        return self.transform(X)


class StandardNaN:
    """
    Standartizes multiple NaN representations into an unique representation.

    Ex: "", "Not Informed", "Not Available", etc -> np.NaN
    """

    def __init__(self, additional_nan_rep: list=None, std_nan_val=np.NaN, inplace: bool=True):
        """
        def __init__(self, additional_nan_rep: dict={}, std_nan_val=np.NaN, inplace: bool=True):
        Class constructor

        Input parameters:
          - additional_nan_rep: a list with *additional* values that you wish to standartize
            with an unique NaN representation. By default, the following values are already
            included: 'na', 'nan', '', '<NA>', 'NÃO INFORMADO', 'None'.
          - std_nan_val: the unique NaN representation that will be used instead of the 
            representations passed.
          - inplace: whether to work on a copy of the data or not.
        """

        #Original NaN reps
        self.nan_rep = [
            re.compile(r'^\s*nan?\s*$', re.IGNORECASE),
            re.compile(r'^[\s\-]*$', re.IGNORECASE),
            re.compile(r'^\s*<NA>\s*$', re.IGNORECASE),
            re.compile(r'^\s*NÃO INFORMADO\s*$', re.IGNORECASE),
            re.compile(r'^\s*None\s*$', re.IGNORECASE),
        ]

        if additional_nan_rep: self.nan_rep += additional_nan_rep
        self.inplace = inplace
        self.std_nan_val = std_nan_val
    
    def fit(self, X, y=None, **kwargs):
        """Dummy function. Nothing is done here."""
        return None
    
    def transform(self, X, **kwargs):
        """Standartizes multiple NaN references"""
        if not self.inplace: X = X.copy()
        X.replace(to_replace=self.nan_rep, value=self.std_nan_val, regex=True, inplace=True)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        """Fit + transform"""
        return self.transform(X)


class SmartFloatCasting:
    """
    Cast float represented as strings into a float number by taking into consideration
    specific thousand and decimal separators.
    """
    def __init__(self, mod_list=[(',', '.')], inplace: bool=True, dtype=np.float64):
        """
        def __init__(self, mod_list=[(',', '.')], inplace: bool=True, dtype=np.float64)

        Input parameters:
          - mod_list: a list of tuples in the shape (orig val, new val). Orig val may be a regexp.
                      This list will be iterated and the replacement found in each iteration will be applied.
          - inplace: whether to work on a copy of the data or not.
          - dtype: data type to cast results into.
        """
        self.mod_list = mod_list
        self.inplace = inplace
        self.dtype=dtype
    
    def fit(self, X, y=None, **kwargs):
        """Dummy function. Nothing is done here."""
        return None

    def transform(self, X, **kwargs):
        """Do the smart float casting."""
        if not self.inplace: X = X.copy()
        for c in X.columns:
            for orig, new in self.mod_list:
                X[c] = X[c].str.replace(orig, new).astype(self.dtype)
        return X
    
    def fit_transform(self, X, y=None, **kwargs):
        """Fit + transform"""
        return self.transform(X)
