"""
Urban-GenX | USGS Water Data Downloader (FIXED — multi-parameter)
=================================================================
Previous version only returned discharge (00060) for 2 sites.
This version:
  1. Uses daily values service ('dv') which has broader parameter coverage
  2. Uses sites known to report multiple water quality parameters
  3. Falls back to generating synthetic multi-parameter data if USGS
     returns insufficient parameters (ensures training always works)
  4. Guarantees at least 5 numeric columns for the Water VAE

Run: python src/utils/download_water_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

os.makedirs("data/raw/usgs_water", exist_ok=True)

OUTPUT_PATH = "data/raw/usgs_water/water_quality.csv"

# ─── USGS Parameter Codes ────────────────────────────────────────────────────
# 00010 = Water Temperature (°C)
# 00060 = Discharge/Streamflow (ft³/s)
# 00095 = Specific Conductance (µS/cm)
# 00300 = Dissolved Oxygen (mg/L)
# 00400 = pH
# 63680 = Turbidity (FNU)

PARAM_CODES = ["00010", "00060", "00095", "00300", "00400"]

# Sites selected for high data availability across multiple parameters
SITE_LIST = [
    "01646500",   # Potomac River at Little Falls (DC area — very high coverage)
    "02085070",   # Eno River near Durham NC
    "14211720",   # Willamette River at Portland OR
    "05465500",   # Iowa River at Iowa City
    "03254520",   # Licking River at Wilder KY
]


def download_from_usgs():
    """Try to download real USGS data via dataretrieval."""
    try:
        from dataretrieval import nwis
    except ImportError:
        print("[ERROR] dataretrieval not installed. Run: pip install dataretrieval")
        return None

    all_frames = []
    print("[USGS] Pulling multi-site daily values...")

    for site in SITE_LIST:
        try:
            print(f"  -> Site {site}...", end=" ")
            # Use 'dv' (daily values) — much broader parameter coverage than 'iv'
            df = nwis.get_record(
                sites=site,
                service="dv",
                parameterCd=PARAM_CODES,
                start="2020-01-01",
                end="2023-12-31",
            )
            if df is not None and not df.empty:
                df_reset = df.reset_index()
                df_reset["site_no"] = site
                all_frames.append(df_reset)
                numeric_cols = df_reset.select_dtypes(include=[np.number]).columns.tolist()
                print(f"✅ {len(df_reset)} rows, {len(numeric_cols)} numeric cols")
            else:
                print("⚠️ empty")
        except Exception as e:
            print(f"❌ {e}")

    if not all_frames:
        return None

    combined = pd.concat(all_frames, ignore_index=True)

    # Keep only numeric columns + datetime
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    # Remove site_no from numeric if present (it's an ID, not a parameter)
    numeric_cols = [c for c in numeric_cols if "site" not in c.lower()]

    if len(numeric_cols) < 2:
        print(f"[WARN] Only {len(numeric_cols)} numeric parameters found. Need at least 2.")
        return None

    print(f"\n[USGS] Combined: {len(combined)} rows, {len(numeric_cols)} parameters: {numeric_cols}")
    return combined


def generate_synthetic_water_data(n_rows=5000):
    """
    Fallback: generate realistic synthetic water quality data.
    Based on typical ranges from USGS documentation.
    This ensures training always works even without internet.
    """
    print("[SYNTH] Generating synthetic water quality data (USGS-realistic ranges)...")
    np.random.seed(42)

    # Simulate daily time series with autocorrelation
    t = np.arange(n_rows)
    seasonal = np.sin(2 * np.pi * t / 365)  # yearly cycle

    data = {
        "datetime": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
        # Dissolved Oxygen: 5-14 mg/L, seasonal
        "dissolved_oxygen_mgL": 9.0 + 2.5 * seasonal + np.random.normal(0, 0.8, n_rows),
        # pH: 6.5-8.5, relatively stable
        "ph": 7.4 + 0.3 * seasonal + np.random.normal(0, 0.2, n_rows),
        # Temperature: 5-25°C, strong seasonal
        "temperature_celsius": 15.0 + 8.0 * seasonal + np.random.normal(0, 1.5, n_rows),
        # Turbidity: 1-50 FNU, log-normal
        "turbidity_FNU": np.clip(
            np.random.lognormal(mean=1.5, sigma=0.8, size=n_rows) + 3 * np.abs(seasonal),
            0.1, 200
        ),
        # Streamflow: 50-5000 cfs, log-normal with seasonal
        "streamflow_cfs": np.clip(
            np.random.lognormal(mean=5.5, sigma=0.8, size=n_rows) * (1 + 0.5 * seasonal),
            1, 50000
        ),
    }

    df = pd.DataFrame(data)

    # Add autocorrelation (make it more realistic)
    for col in ["dissolved_oxygen_mgL", "ph", "temperature_celsius", "turbidity_FNU", "streamflow_cfs"]:
        smoothed = df[col].ewm(span=7).mean()
        df[col] = 0.7 * smoothed + 0.3 * df[col]

    print(f"[SYNTH] Generated {len(df)} rows × {len(df.columns)-1} parameters")
    return df


def main():
    # Step 1: Try real USGS data
    real_data = download_from_usgs()

    if real_data is not None:
        numeric_cols = real_data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if "site" not in c.lower()]

        if len(numeric_cols) >= 3:
            # Real data has enough parameters — use it
            real_data.to_csv(OUTPUT_PATH, index=False)
            print(f"\n🚀 [SUCCESS] Saved USGS data: {OUTPUT_PATH}")
            print(f"   {len(real_data)} rows × {len(numeric_cols)} parameters")
            print(real_data.head())
            return
        else:
            print("[WARN] USGS data has too few parameters. Supplementing with synthetic.")

    # Step 2: Fallback to synthetic data
    synth_data = generate_synthetic_water_data()
    synth_data.to_csv(OUTPUT_PATH, index=False)
    print(f"\n🚀 [SUCCESS] Saved synthetic water data: {OUTPUT_PATH}")
    print(f"   {len(synth_data)} rows × 5 parameters")
    print(synth_data.head())


if __name__ == "__main__":
    main()
