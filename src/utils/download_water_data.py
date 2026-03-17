"""
Urban-GenX | USGS Water Data Downloader (Fixed Unpacking Error)
Focusing on high-availability parameters for the Water VAE Node.
"""

from dataretrieval import nwis
import pandas as pd
import os

os.makedirs("data/raw/usgs_water", exist_ok=True)

# 11102300: Santa Clara River (Reliable for pH/Temp)
# 11087020: LA River
site_list = ["11102300", "11087020", "11094350"]
# 00010 = Temp, 00400 = pH, 00060 = Discharge
param_codes = ["00010", "00400", "00060"]

all_data = []

print("[USGS] Pulling multi-site water data...")

for site in site_list:
    try:
        print(f"-> Querying Site: {site}")
        
        # FIXED: get_record for 'iv' returns only 'df', not 'df, meta'
        df = nwis.get_record(
            sites=site, 
            service='iv', 
            parameterCd=param_codes, 
            start="2023-01-01", 
            end="2024-01-01"
        )
        
        if df is not None and not df.empty:
            print(f"   ✅ Found {len(df)} records for {site}")
            all_data.append(df.reset_index())
        else:
            print(f"   ⚠️ No 'iv' data for {site} in this range.")
            
    except Exception as e:
        print(f"   ❌ Skip {site}: {e}")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    output_path = "data/raw/usgs_water/water_quality.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n🚀 [USGS] Success! Master file created: {output_path}")
    print(final_df.head())
else:
    print("\n💀 [USGS] Error: All sites returned empty. Ensure you have an internet connection.")