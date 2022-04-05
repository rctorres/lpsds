import io
import os
import pandas as pd
import s3fs
from contextlib import contextmanager


class FileHandler():
    def __init__(self, bucket):
        self.bucket = bucket
    def get_full_path(self, dir_name, prefix='', fname=''):
        raise NotImplemented
    def fopen(self, fname, mode):
        raise NotImplemented
        
        
        
class S3FileHandler(FileHandler):
    def __init__(self, bucket):
        super().__init__(bucket)

    def get_full_path(self, dir_name='', prefix='', fname=''):
        return os.path.join(f's3://{self.bucket}', dir_name, prefix, fname)
    
    @contextmanager
    def fopen(self, fname, mode):
        fs = s3fs.S3FileSystem()
        fobj = fs.open(fname, mode)
        yield fobj
        fobj.close()

        
class LocalFileHandler(FileHandler):    
    def __init__(self, bucket):
        super().__init__(bucket)

    def get_full_path(self, dir_name='', prefix='', fname=''):
        """
        get_full_path(dir_name, prefix='', fname='')

        Creates a file path as dir_"name/prefix/fname".
        Creates "dir_name/prefix" if it does not exists.
        Returns: "dir_name/prefix/fname"
        """
        base_path =  os.path.join(self.bucket, dir_name, prefix)

        #If both dir_name and prefix are empty, we can assume
        #fname contains the full path.
        if (dir_name == '') and (prefix == ''):
            dir_name, fname = os.path.split(fname)

        #Creating full absolute path.
        path = os.path.abspath(os.path.join(self.bucket, dir_name, prefix))
        #If the path does not exist, try using relative path instead
        if not os.path.exists(path): path = os.path.join(self.bucket, dir_name, prefix)
        #If the path still does not exist, we create it
        if not os.path.exists(path): os.makedirs(path)
        #Appending the file name (if existent) and returning.
        return os.path.join(path, fname)
    
    @contextmanager
    def fopen(self, fname, mode):
        f = open(fname, mode)
        yield f
        f.close()
