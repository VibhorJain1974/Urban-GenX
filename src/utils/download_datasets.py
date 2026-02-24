"""
Urban-GenX | Automated Dataset Downloader
Run once: python src/utils/download_datasets.py
Downloads: UrbanSound8K, METR-LA (via Kaggle), USGS Water Quality
NOTE: Cityscapes requires manual registration at cityscapes-dataset.com
"""

import os
import subprocess
import sys

def make_dirs():
    for d in [
        "data/raw/cityscapes",
        "data/raw/urbansound8k",
        "data/raw/metr-la",
        "data/raw/usgs_water",
    ]:
        os.makedirs(d, exist_ok=True)
    print("[✅] Directory structure created.")

def download_urbansound8k():
    print("\n[📥] Downloading UrbanSound8K via soundata...")
    try:
        import soundata
        dataset = soundata.initialize('urbansound8k', data_home='data/raw/urbansound8k')
        dataset.download()
        dataset.validate()
        print("[✅] UrbanSound8K downloaded and validated.")
    except ImportError:
        print("[⚠️] soundata not installed. Running: pip install soundata")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundata"])
        import soundata
        dataset = soundata.initialize('urbansound8k', data_home='data/raw/urbansound8k')
        dataset.download()

def download_metr_la():
    print("\n[📥] Downloading METR-LA via Kaggle CLI...")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "annnnguyen/metr-la-dataset",
             "-p", "data/raw/metr-la",
             "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[✅] METR-LA downloaded.")
        else:
            print(f"[⚠️] Kaggle CLI error: {result.stderr}")
            print("[ℹ️] Manual download: https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset")
    except FileNotFoundError:
        print("[⚠️] Kaggle CLI not found.")
        print("     Install: pip install kaggle")
        print("     Then place kaggle.json in ~/.kaggle/")
        print("     Manual: https://data.mendeley.com/datasets/s42kkc5hsw")

def download_usgs_water():
    print("\n[📥] Downloading USGS Water Quality via dataretrieval...")
    try:
        import dataretrieval.nwis as nwis
        df, _ = nwis.get_qwdata(
            sites=["11087020", "11098000", "11119750"],
            parameterCd=["00300", "00400", "00010", "63680"],
            start="2010-01-01",
            end="2020-12-31",
        )
        df.to_csv("data/raw/usgs_water/water_quality.csv")
        print(f"[✅] USGS data saved: {len(df)} records.")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dataretrieval"])
        download_usgs_water()

def print_cityscapes_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  CITYSCAPES — Manual Download Required                       ║
║                                                                  ║
║  1. Register at: https://www.cityscapes-dataset.com/register/   ║
║     (Use your university email — approved instantly)            ║
║                                                                  ║
║  2. Download these 2 files:                                      ║
║     → leftImg8bit_trainvaltest.zip  (~11 GB)                    ║
║     → gtFine_trainvaltest.zip       (~241 MB)                   ║
║                                                                  ║
║  3. Extract both to: data/raw/cityscapes/                        ║
║                                                                  ║
║  OR use Kaggle (no registration):                                ║
║  kaggle datasets download -d electraawais/cityscape-dataset     ║
║                       -p data/raw/cityscapes --unzip             ║
╚══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print("=" * 60)
    print("  Urban-GenX | Dataset Downloader")
    print("=" * 60)
    make_dirs()
    download_urbansound8k()
    download_metr_la()
    download_usgs_water()
    print_cityscapes_instructions()
    print("\n[🏙️] All automated downloads complete!")
    print("     Place Cityscapes manually when ready.")
