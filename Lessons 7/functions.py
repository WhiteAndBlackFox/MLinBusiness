# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series

def series_factorizer(series: Series):
    series, unique = pd.factorize(series)
    reference = {x: i for x, i in enumerate(unique)}
    print(reference)
    return series, reference
