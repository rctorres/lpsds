import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def setup(width=14., num_cols=None, num_rows=None, float_fmt='{:,.3f}'):
    mpl.rcParams['figure.figsize'] = [width, width/1.5]
    sns.set(context='talk', style='white', palette='muted')
    pd.options.display.max_columns =  num_cols
    pd.options.display.max_rows =  num_rows
    pd.options.display.float_format = float_fmt.format


setup()
