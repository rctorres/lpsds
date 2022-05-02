import pytest

from preprocessing import standard_nan
import pandas as pd
import numpy as np


class TestStandardNaN:

    def test_case_insensitiveness(self):

        df = pd.DataFrame({'a' : ['na', 'nA', 'NA'], 
            'b' : ['nan', 'Nan', 'NAN'],
            'c' : ['', ' ', '   '],
            'd' : ['<NA>', '<na>', '<Na>'],
            'e' : ['não informado', 'Não Informado', 'Não InFoRmAdO']})
        res = standard_nan(df)

        for idx, c in df.iterrows:
            for col in df.columns:
                assert c[col] == np.nan
