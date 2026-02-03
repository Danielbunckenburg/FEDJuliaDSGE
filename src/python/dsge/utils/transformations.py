import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter as sm_hpfilter

def annual_to_quarter(v):
    return v / 4.0

def quarter_to_annual(v):
    return v * 4.0

def quarter_to_annual_percent(v):
    return v * 400.0

def nominal_to_real(col, df, deflator_mnemonic='GDPDEF'):
    return df[col] / df[deflator_mnemonic]

def percapita(col, df, population_mnemonic):
    return df[col] / df[population_mnemonic]

def difflog(x):
    """
    Computes the first difference of the log of x.
    First element is NaN to maintain array length.
    """
    if isinstance(x, pd.Series):
        return np.log(x).diff()
    x = np.array(x)
    res = np.empty_like(x, dtype=float)
    res[0] = np.nan
    res[1:] = np.log(x[1:]) - np.log(x[:-1])
    return res

def one_qtr_pct_change(y):
    return 100.0 * difflog(y)

def loggrowthtopct(y):
    return 100.0 * (np.exp(y / 100.0) - 1.0)

def loggrowthtopct_annualized(y):
    return 100.0 * (np.exp(y / 100.0)**4 - 1.0)

def loggrowthtopct_annualized_percapita(y, pop_growth):
    return 100.0 * (np.exp(y / 100.0 + pop_growth)**4 - 1.0)

def hpfilter(y, lamb):
    """
    Applies Hodrick-Prescott filter.
    Returns (trend, cycle).
    """
    cycle, trend = sm_hpfilter(y, lamb)
    return trend, cycle

def quarter_to_date(q_str):
    """
    Converts 'YYYYqX' or 'YYYY-qX' to end of quarter date.
    """
    q_str = q_str.upper()
    if '-' in q_str:
        year, q = q_str.split('-Q')
    else:
        year = q_str[:4]
        q = q_str[4:] if 'Q' not in q_str else q_str.split('Q')[1]
    
    month = 3 * int(q)
    return pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.QuarterEnd(0)
