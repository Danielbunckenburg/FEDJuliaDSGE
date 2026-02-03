import pandas as pd
import numpy as np
import os

def load_csv_data(filepath, model):
    """
    Loads data from a CSV file.
    The CSV should have a header row with observable names (e.g., obs_gdp, obs_hours...).
    """
    df = pd.read_csv(filepath)
    
    # Ensure all model observables are present
    missing = [obs for obs in model.observables.keys() if obs not in df.columns]
    if missing:
        print(f"Warning: Missing observables in CSV: {missing}. Filling with NaNs.")
        for m in missing:
            df[m] = np.nan
            
    # Reorder columns to match model.observables
    data_matrix = df[list(model.observables.keys())].values.T
    
    return data_matrix

def create_data_template(model, filepath, T=40):
    """
    Creates a template CSV file for the user to fill.
    """
    df = pd.DataFrame(np.nan, index=range(T), columns=list(model.observables.keys()))
    # Add a date column if missing
    if "date" not in df.columns:
        df.insert(0, "date", pd.date_range("2010-01-01", periods=T, freq="Q"))
        
    df.to_csv(filepath, index=False)
    print(f"Created data template at {filepath}")
