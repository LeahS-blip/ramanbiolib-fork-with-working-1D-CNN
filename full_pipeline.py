import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Optional

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# allow imports from scripts folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from fetch_raman_sdbs_advanced import RamanSpectraFetcher  # type: ignore  (local script)

# --- CONFIGURATION ---

INPUT_CSV = "molecules.csv"
RAW_DATA_DIR = Path("data/Raman")
PROCESSED_DIR = Path("data/processed")
MERGED_DIR = Path("data/merged")
LOG_FILE = Path("logs/sdbs_fetch.log")

WAV_GRID = np.linspace(400, 1800, 1351)

USE_SAVGOL = True
SAVGOL_WINDOW = 11
SAVGOL_POLY = 2

USE_DERIVATIVE = True
NORMALIZATION = "area"  # 'area', 'min-max', 'z-score', 'none'

# Ensure folders exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MERGED_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Initialize SDBS fetcher
fetcher = RamanSpectraFetcher(str(RAW_DATA_DIR))

# --- HELPER FUNCTIONS ---

# Note: full_pipeline no longer directly queries SDBS; the fetcher
# handles searching, downloading and conversion.  Each molecule will be
# saved as a JCAMP file plus a CSV under RAW_DATA_DIR/<class>.


def read_converted_csv(cls: str, molecule_name: str):
    """Return (wn, intensity) arrays from the CSV produced by the fetcher.

    The fetcher converts a downloaded JCAMP into a two-column CSV where
    the first column is wavenumber and the second is intensity.  File
    naming follows the same logic used in ``RamanSpectraFetcher.process_molecule``.
    """
    fname = molecule_name.lower().replace(" ", "_") + ".csv"
    path = RAW_DATA_DIR / cls / fname
    if not path.exists():
        return None, None
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1 or data.shape[1] < 2:
        return None, None
    return data[:, 0], data[:, 1]



def preprocess_spectrum(wn: np.ndarray, intensity: np.ndarray):
    """
    Resample + normalize + optional smoothing + optional derivative channels.
    Returns an array of shape (channels, GRID_POINTS).
    """
    # Interpolate to a fixed common grid
    interp_func = interp1d(wn, intensity, kind="linear", bounds_error=False, fill_value=0.0)
    y_resampled = interp_func(WAV_GRID)

    # Normalize
    if NORMALIZATION == "area":
        total = np.sum(y_resampled) + 1e-12
        y_resampled = y_resampled / total
    elif NORMALIZATION == "min-max":
        y_min, y_max = y_resampled.min(), y_resampled.max()
        y_resampled = (y_resampled - y_min) / (y_max - y_min + 1e-12)
    elif NORMALIZATION == "z-score":
        y_resampled = (y_resampled - y_resampled.mean()) / (y_resampled.std() + 1e-12)

    # Savitzky-Golay smoothing
    if USE_SAVGOL:
        y_resampled = savgol_filter(y_resampled, SAVGOL_WINDOW, SAVGOL_POLY)

    # Optionally add derivative channel
    if USE_DERIVATIVE:
        dydx = np.gradient(y_resampled)
        return np.vstack([y_resampled, dydx])

    return y_resampled[np.newaxis, :]

# --- MAIN PIPELINE ---

def main(input_csv: str):
    # first, read list and fetch raw spectra via the fetcher
    df = pd.read_csv(input_csv)
    logger.info(f"Starting download of {len(df)} molecules from SDBS...")

    if not fetcher.process_csv(input_csv):
        logger.error("Failed to download spectra; aborting pipeline.")
        return

    classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_list, y_list = [], []
    missing = []

    # Process each molecule: convert already-downloaded CSV to numpy arrays
    for _, row in df.iterrows():
        name = row["molecule_name"].strip()
        cls = row["class"].strip()

        out_cls_dir = PROCESSED_DIR / cls
        out_cls_dir.mkdir(exist_ok=True, parents=True)

        wn, intensity = read_converted_csv(cls, name)
        if wn is None:
            missing.append(name)
            with open(LOG_FILE, "a") as lf:
                lf.write(f"{name}\n")
            continue

        processed = preprocess_spectrum(wn, intensity)

        # Save per-molecule processed CSV (in processed/ rather than raw dir)
        csv_path = out_cls_dir / f"{name.replace(' ', '_')}.csv"
        np.savetxt(csv_path, processed.T, delimiter=",")

        X_list.append(processed)
        y_list.append(class_to_idx[cls])

    # Merge into dataset
    if not X_list:
        logger.warning("No spectra were processed; X/y arrays would be empty."
                       " Check the fetch log or input CSV.")
        return

    X = np.stack(X_list)
    y = np.array(y_list)

    np.save(MERGED_DIR / "X.npy", X)
    np.save(MERGED_DIR / "y.npy", y)
    with open(MERGED_DIR / "classes.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # Print summary
    print("---------- Pipeline Complete ----------")
    print(f"Total spectra processed: {len(y)}")
    for cls, idx in class_to_idx.items():
        print(f"{cls}: {(y == idx).sum()} spectra")
    print("Saved:")
    print(f"  X.npy -> {MERGED_DIR / 'X.npy'}")
    print(f"  y.npy -> {MERGED_DIR / 'y.npy'}")
    print(f"  classes.json -> {MERGED_DIR / 'classes.json'}")
    print(f"Missing spectra logged to {LOG_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch + preprocess Raman spectra from SDBS into a labeled dataset")
    parser.add_argument("--input_csv", required=True, help="CSV with columns molecule_name,class")
    args = parser.parse_args()
    main(args.input_csv)