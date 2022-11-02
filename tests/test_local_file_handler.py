import os
import pytest
import lpsds.handlers

def fake_make_dir(path):
    return path

@pytest.fixture(autouse=True)
def set_mocks(monkeypatch):
    monkeypatch.setattr(os, "makedirs", fake_make_dir)

class TestGetFullPath():
    @pytest.fixture
    def bucket(self):
        return '/bucket'
    
    @pytest.fixture
    def fh_obj(self, bucket):
        return lpsds.handlers.LocalFileHandler(bucket=bucket)

    def test_full_parameters_setting(self, bucket, fh_obj):
        dir_name = 'directory'
        prefix = 'prefix'
        fname = 'filename.ext'
        
        ret = fh_obj.get_full_path(dir_name=dir_name, prefix=prefix, fname=fname)
        assert ret == os.path.join(bucket, dir_name, prefix, fname)

    def test_full_path_setting(self, bucket, fh_obj):
        fname = 'directory/prefix/filename.ext'
        ret = fh_obj.get_full_path(fname=fname)
        assert ret == os.path.join(bucket, fname)
 