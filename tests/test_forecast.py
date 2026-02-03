import numpy as np
import pandas as pd
from dsge.models.m1002 import Model1002
from dsge.forecast import forecast

def test_forecast_logic():
    print("Initializing Model1002...")
    m = Model1002()
    TTT, RRR, CCC = m.solve()
    
    # Simulate some data
    n_obs = m.n_observables
    T = 50
    data = np.random.randn(n_obs, T)
    
    print("Running Filter...")
    filt_res = m.filter(data, outputs=['states'])
    s_final = filt_res[:, -1]
    
    print("Running Forecast (12 quarters)...")
    forecast_states, forecast_obs = forecast(m, TTT, RRR, CCC, s_final, horizon=12)
    
    print(f"Forecast shape: {forecast_obs.shape}")
    
    # Convert to DataFrame for readability
    df_forecast = pd.DataFrame(forecast_obs, columns=list(m.observables.keys()))
    print("\nForecasted Observables (First 3 qtrs):")
    print(df_forecast.head(3))
    
    print("\nRunning Kalman Smoother...")
    s_smooth, P_smooth = m.smooth(data)
    print(f"Smoothed states shape: {s_smooth.shape}")
    
    # The last smoothed state should be the same as the last filtered state (by theory)
    diff = np.abs(s_smooth[:, -1] - s_final).max()
    print(f"Max diff between last filtered and smoothed state: {diff:.2e}")
    assert diff < 1e-10, "Smoother logic failure"
    
    print("\nForecast and Smoother test passed!")

if __name__ == "__main__":
    test_forecast_logic()
