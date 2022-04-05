"""
Handles data acquisition, loading and saving operations.
"""

import sys
import os
import pandas as pd
from .handlers import LocalFileHandler, S3FileHandler
import pickle

HANDLER_NAME = os.environ.get('LPSDS_HANDLER', 'LocalFileHandler')
BUCKET = os.environ.get('LPSDS_BUCKET')
HANDLER = getattr(sys.modules[__name__], HANDLER_NAME)(BUCKET)

def get_path(dir_name='', prefix='', fname=''):
    """
    def get_path(dir_name='', prefix='', fname='')

    Returns the full path for a given file or a given path (if filename is omitted)
    taking into consideration the handler being used.

    Input parameters:
      - dir_name: base directory.
      - prefix: subdirectory under "dir_name" where the image file will be saved.
      - fname: file name.
    
    Return: the location / filename full path.

    """
    return HANDLER.get_full_path(dir_name, prefix, fname)


def read_fig(full_path_name):
    with HANDLER.fopen(full_path_name, 'rb') as image_content:
        return image_content.read()


def to_fig(fig, full_path_name, tight=True, remove_title=False):
    """
    to_fig(fig, fname, case_name='', dir_name='', tight=True, remove_title=False)

    Saves a matplotlib compatible plot in "dir_name/prefix/fname.pdf"
    Input parameters:
      - fig: instance returned by plt.figure().
      - fname: file name (without extensioon) where the image will be saved.
      - prefix: subdirectory under "dir_name" where the image file will be saved.
      - dir_name: base directory for saving the image.
      - tight: enforces matplotlib tight laypout.
      - remove_title: removes title and suptitle before saving the file (useful for papers and articles).
    """
    if tight: fig.tight_layout()
    if remove_title:
        fig.title(' ')
        fig.suptitle(' ')
    
    fname, _ = os.path.splitext(full_path_name)
    
    fmt_map = {
        'pdf' : {},
        'png' : dict(dpi=300),
    }
    
    for fmt, opt in fmt_map.items():
        full_name = f'{fname}.{fmt}'    
        with HANDLER.fopen(full_name , 'wb') as f:
            fig.savefig(f, format=fmt, **opt)


def read_pickle(full_path_name):
    with HANDLER.fopen(full_path_name, 'rb') as f:
        return pickle.load(f)

    
def to_pickle(obj, full_path_name):
    with HANDLER.fopen(full_path_name, 'wb') as f:
        pickle.dump(obj, f)
    

def to_excel(df, full_path_name, **kwargs):
    """
   Saves an Excel data file to "full_path_name".
    Input parameters:
      - full_path_name: The full path to the file to be generated.
      - **kwargs: to be passed to pandas.to_excel
    """
    with HANDLER.fopen(full_path_name, 'wb') as f:
        df.to_excel(f, **kwargs)


def get_function(pandas_ref, fname, operation):
    assert operation in ['read', 'to'], 'Operation must be either "read" or "to"'

    #We will employ the file name extension to derive the filetype we wish to save
    #and the function that should handle it.
    _, ext = os.path.splitext(fname)
    ext = ext[1:] # Removing the dot from the extension.

    format = None
    pandas_function = True
    if ext in ['xls', 'xlsx']:
        #If we are reading an excel file, we can use pd.read_excel.
        #Otherwise, we must use a custom version to handle S3 writting.
        pandas_function = operation == 'read'
        format = 'excel'
    elif ext in ['pdf', 'png', 'fig']:
        format = 'fig'
        pandas_function = False
    elif ext == 'pickle':
        format = 'pickle'
        pandas_function = False        
    else: format = ext

    #Here we derive whether we want to save or load in the provide file format.
    function_name = f'{operation}_{format}'

    #Getting a reference to the object from which the function we will call
    #to handle the oepration (pd.to_excel, to_fig, pf.read_parquet, etc.)
    obj_ref = pandas_ref if pandas_function else sys.modules[__name__]

    #Getting the function that will manage the content.
    func = getattr(obj_ref, function_name)

    #Getting the right wrapper. We need this wrapper to make sure
    #we will have the same interface between all saving possibilities.
    wrapper_func = save_with_pandas if pandas_function else save_without_pandas
    return func, wrapper_func


def save_with_pandas(df, full_name, func, **kwargs):
    func(full_name, **kwargs)


def save_without_pandas(df, full_name, func, **kwargs):
    func(df, full_name, **kwargs)


def save(df, fname, dir_name='', prefix='', **kwargs):
    func, wrapper_func = get_function(df, fname, 'to')
    full_name = HANDLER.get_full_path(dir_name, prefix, fname)
    wrapper_func(df, full_name, func, **kwargs)


def load(fname, dir_name='', prefix='', **kwargs):
    func, _ = get_function(pd, fname, 'read')
    file_obj = HANDLER.get_full_path(dir_name, prefix, fname)
    ret = func(file_obj, **kwargs)
    return ret
