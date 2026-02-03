import pandas as pd
import numpy as np
import requests
from datetime import datetime
from collections import OrderedDict

class Observable:
    def __init__(self, key, mnemonics, fwd_transform, rev_transform, 
                 name="", description=""):
        self.key = key
        self.mnemonics = mnemonics # List of FRED mnemonics
        self.fwd_transform = fwd_transform
        self.rev_transform = rev_transform
        self.name = name
        self.description = description

class FRED:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/"

    def get_series(self, series_id, start_date=None, end_date=None):
        if not self.api_key:
            raise ValueError("FRED API Key required")
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "frequency": "q"
        }
        if start_date: params["observation_start"] = start_date
        if end_date: params["observation_end"] = end_date
        
        response = requests.get(self.base_url + "series/observations", params=params)
        response.raise_for_status()
        data = response.json()["observations"]
        
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")["value"]

def load_data(m, start_date, end_date, fred_api_key=None):
    """
    Loads data for model m from FRED or local CSV.
    """
    mnemonics = set()
    for obs in m.observable_mappings.values():
        for mnem in obs.mnemonics:
            if mnem.endswith("__FRED"):
                mnemonics.add(mnem.replace("__FRED", ""))
    
    # Simple fetcher logic
    fred = FRED(api_key=fred_api_key)
    dfs = []
    for mnem in mnemonics:
        try:
            series = fred.get_series(mnem, start_date, end_date)
            series.name = mnem
            dfs.append(series)
        except Exception as e:
            print(f"Warning: Could not fetch {mnem}: {e}")
            
    if not dfs:
        return pd.DataFrame(columns=["date"])
        
    df = pd.concat(dfs, axis=1).reset_index()
    return df

def transform_data(m, levels):
    """
    Applies forward transforms to levels data to create observables.
    """
    data = pd.DataFrame({"date": levels["date"]})
    for key, obs in m.observable_mappings.items():
        try:
            # The fwd_transform expects the full levels DataFrame
            data[key] = obs.fwd_transform(levels)
        except Exception as e:
            print(f"Warning: Could not transform {key}: {e}")
            data[key] = np.nan
    return data
