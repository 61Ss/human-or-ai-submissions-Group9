"""Data loading helpers shared by all experiments."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
SPLITS = ("train", "unlabeled", "test")


def load_split(name, data_dir=DATA_DIR):
    return pd.read_csv(Path(data_dir) / f"{name}.csv")


def load_all(data_dir=DATA_DIR):
    """Return one DataFrame with columns id, text, label (NaN if hidden), split."""
    frames = []
    for name in SPLITS:
        df = load_split(name, data_dir)
        df["split"] = name
        if "label" not in df:
            df["label"] = float("nan")
        frames.append(df[["id", "text", "label", "split"]])
    return pd.concat(frames, ignore_index=True)
