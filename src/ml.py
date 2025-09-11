from __future__ import annotations
import numpy as np
import pandas as pd

def add_quick_prob(
    df: pd.DataFrame,
    w_sent: float = 0.6,
    w_mom: float = 0.4,
    mom_scale: float = 10.0,
    temp: float = 3.0
) -> pd.DataFrame:
    """
    Adds a 'p_up' column using a simple blend:
        z = w_sent * sent_mean + w_mom * tanh(mom_scale * ret_5)
        p_up = sigmoid(temp * z)
    Returns a copy of df with p_up in [0, 1]
    """
    if df is None or df.empty:
        return df.assign(p_up=pd.Series(dtype="float64"))
    
    out = df.copy()
    
    # ensure numeric
    sent = out["sent_mean"].astype(float)
    mom = out["ret_5"].astype(float)
    
    z = w_sent * sent + w_mom * np.tanh(mom_scale * mom)
    out["p_up"] = 1.0 / (1.0 + np.exp(-temp * z))
    out["p_up"] = out["p_up"].clip(0.0, 1.0)
    return out