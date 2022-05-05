import collections.abc

class ObjectView(dict):
    """Data structure that improves a regular map structure so
    its components can be accessed as map['value'] or map.value.
    It works only for string keys.
    """
    def __getattr__(self, name):
        if name in self: return self[name]
        else: raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self: del self[name]
        else: raise AttributeError("No such attribute: " + name)


def to_list(val):
    """
    If val is an iterable collection other than a string, returns val,
    otherwise, returs [val].
    """
    if isinstance(val, collections.abc.Iterable) and type(val) is not str:
        return val
    else: return [val]


def smart_tuple(vec):
    """
    If len(vec) == 1, returns vec[0], otherwise, returns vec as a tuple.
    """
    if len(vec) == 1: return vec[0]
    else: return tuple(vec)


def keep(df, index=None, columns=None, inplace=True):
    """
    def keep(index=None, columns=None, inplace=True)
    
    The opposite of pd.drop function. In this function, you specify the index / columns you
    want to keep. I.e. the function will drop everything that does not match the index/cools
    specified.
    
    IMPORTANT REMARKS:
     - differently than the drop function, "inplace" optioon defaults to True.
     - index and columns are FILTER, not the final list of values to be kept.
    
    Returns: the filtered dataframe if inplace=True or None otherwise.
    """
    
    if index is not None:
        index = df.loc[~index].index
    
    if columns is not None:
        columns = df.columns[~columns]
    
    return df.drop(index=index, columns=columns, inplace=inplace)
