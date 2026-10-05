"""Data utilities for the FMA-small genre classification task."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_small_labels(metadata_dir):
    """Return a DataFrame (index=track_id, column=genre) for the fma_small subset."""
    tracks = pd.read_csv(
        os.path.join(metadata_dir, "tracks.csv"), index_col=0, header=[0, 1]
    )
    small = tracks[tracks["set", "subset"] == "small"]
    df = pd.DataFrame({"genre": small["track", "genre_top"]})
    df.index.name = "track_id"
    return df


def audio_path(audio_dir, track_id):
    """FMA stores track 2 at fma_small/000/000002.mp3 (folder = first 3 digits)."""
    tid = f"{int(track_id):06d}"
    return os.path.join(audio_dir, tid[:3], tid + ".mp3")


def make_splits(df, seed=42):
    """Stratified 70/20/10 train/val/test split. Adds a 'split' column."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["genre"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=1 / 3, stratify=temp["genre"], random_state=seed
    )
    out = df.copy()
    out.loc[train.index, "split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out

