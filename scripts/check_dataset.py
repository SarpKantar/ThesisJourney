#!/usr/bin/env python
"""Lightweight sanity checks for the CNN Filter DB dataset."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_DATASET = "data/dataset.h5"


def scale_filters(x):
    den = np.abs(x).max(axis=1)
    den = np.where(den == 0, 1, den)[:, None]
    return x / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.environ.get("CNN_FILTER_DB_DATASET", DEFAULT_DATASET),
        help="Path to dataset.h5. Defaults to CNN_FILTER_DB_DATASET or data/dataset.h5.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of filters to use for the PCA smoke test",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Set CNN_FILTER_DB_DATASET or pass --dataset /path/to/dataset.h5."
        )

    print(f"dataset: {dataset_path}")
    print(f"size_gib: {dataset_path.stat().st_size / 1024**3:.2f}")

    with h5py.File(dataset_path, "r") as handle:
        print(f"hdf5_keys: {list(handle.keys())}")
        filters = handle["filters"]
        print(f"filters_shape: {filters.shape}")
        print(f"filters_dtype: {filters.dtype}")

        n = min(args.sample_size, filters.shape[0])
        sample = filters[:n].reshape(n, 9).astype(np.float32)

    meta = pd.read_hdf(dataset_path, "meta")
    print(f"meta_shape: {meta.shape}")
    print(f"meta_index_names: {meta.index.names}")
    print(f"first_filter_ids: {meta.iloc[0]['filter_ids']}")

    sample_scaled = scale_filters(sample)
    pca = PCA(n_components=9)
    transformed = pca.fit_transform(sample_scaled)
    print(f"pca_sample_shape: {transformed.shape}")
    print("pca_explained_variance_ratio:", np.round(pca.explained_variance_ratio_, 6).tolist())
    print("status: ok")


if __name__ == "__main__":
    main()
