"""
Urban-GenX | USGS Water Data Downloader
Downloads dissolved oxygen, pH, temperature, turbidity time series.
Saves to data/raw/usgs_water/water_quality.csv
"""

import dataretrieval.nwis as nwis
import pandas as pd
import os

os.makedirs("data/raw/usgs_water", exist_ok=True)

# ── Example: Los Angeles County water quality sites ──────────────────────────
# Parameter codes:
#   00300 = Dissolved Oxygen (mg/L)
#   00400 = pH
#   00010 = Water Temperature (°C)
#   63680 = Turbidity (FNU)
#   00060 = Streamflow (ft³/s)

sites = [
    "11087020",   # Los Angeles River at Wardlow Road
    "11098000",   # San Gabriel River at Whittier Narrows
    "11119750",   # Santa Clara River near Ventura
]

param_codes = ["00300", "00400", "00010", "63680", "00060"]

print("[USGS] Downloading water quality data...")
df, meta = nwis.get_qwdata(
    sites=sites,
    parameterCd=param_codes,
    start="2010-01-01",
    end="2020-12-31",
)

df.to_csv("data/raw/usgs_water/water_quality.csv")
print(f"[USGS] Saved {len(df)} records → data/raw/usgs_water/water_quality.csv")
print(df.head())
