#!/usr/bin/env python
"""Recompute the full CNN Filter DB PCA basis with bounded memory.

The committed paper notebook performs the following preprocessing before PCA:

1. reshape every 3x3 filter to a row of length nine;
2. cast the raw filter values to ``numpy.float16``;
3. divide each row by its maximum absolute value (leaving zero rows zero);
4. fit a full-rank, nine-component PCA in float64.

This script reproduces those statistics without materializing the roughly
1.46-billion-row matrix.  It merges per-chunk float64 means and centered sums
of products, then eigendecomposes the resulting 9x9 sample covariance matrix.
The sufficient statistics are checkpointed atomically and can be resumed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_DATASET = os.environ.get(
    "CNN_FILTER_DB_DATASET",
    "data/dataset.h5",
)
DEFAULT_OUTPUT = (
    "outputs/(4thEXP)rn18_cifar10_filter_dynamics/"
    "original_cnn_filter_db_pca/cnn_filter_db_full_pca.npz"
)
CHECKPOINT_FORMAT_VERSION = 1
ARTIFACT_FORMAT_VERSION = 1
PREPROCESSING_ID = "cnn_filter_db_paper_float16_maxabs_v1"
PUBLISHED_FULL_DATASET_MEAN = np.asarray(
    [
        -0.04262863,
        -0.04113670,
        -0.04461834,
        -0.04071190,
        -0.03574134,
        -0.04268694,
        -0.04350573,
        -0.04138637,
        -0.04486743,
    ],
    dtype=np.float64,
)

_STOP_SIGNAL: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the original CNN Filter DB nine-dimensional PCA basis using "
            "streaming, resumable covariance statistics."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="CNN Filter DB HDF5 file containing /filters.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Final PCA .npz artifact.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint .npz. Defaults to <output stem>.checkpoint.npz.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000_000,
        help="Number of contiguous filters read per chunk.",
    )
    parser.add_argument(
        "--checkpoint-every-chunks",
        type=int,
        default=5,
        help="Atomically save sufficient statistics after this many chunks.",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=0,
        help="Process only the first N filters; 0 means the full dataset.",
    )
    parser.add_argument(
        "--published-mean-tolerance",
        type=float,
        default=5e-5,
        help="Maximum absolute error allowed against the paper's full-data mean.",
    )
    parser.add_argument(
        "--allow-published-mean-mismatch",
        action="store_true",
        help="Write a full-data artifact even if its mean misses the paper reference.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help="Resume a compatible checkpoint (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Reject an existing checkpoint instead of resuming it.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start from filter zero, replacing any checkpoint and final artifact.",
    )
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=0,
        help=(
            "Stop after N newly processed chunks with a resumable checkpoint; "
            "0 means run to completion. Primarily useful for controlled tests."
        ),
    )
    return parser.parse_args()


@dataclass
class RunningCovariance:
    """Mergeable sample count, mean, and centered sum-of-products matrix."""

    n: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls) -> "RunningCovariance":
        return cls(
            n=0,
            mean=np.zeros(9, dtype=np.float64),
            m2=np.zeros((9, 9), dtype=np.float64),
        )

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != 9 or values.dtype != np.float64:
            raise ValueError(
                f"Expected a float64 (N, 9) chunk, received {values.shape} {values.dtype}"
            )
        batch_n = int(values.shape[0])
        if batch_n == 0:
            return
        batch_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean.copy()
            self.m2 = batch_m2
            return

        old_n = self.n
        total_n = old_n + batch_n
        delta = batch_mean - self.mean
        merge_weight = float(old_n) * (float(batch_n) / float(total_n))
        self.mean = self.mean + delta * (float(batch_n) / float(total_n))
        self.m2 = self.m2 + batch_m2 + np.outer(delta, delta) * merge_weight
        self.n = total_n


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    print(
        f"[signal] Received {signal.Signals(signum).name}; "
        "will checkpoint after the current chunk.",
        flush=True,
    )


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _signal_handler)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def scalar(data: Any, key: str) -> Any:
    value = data[key]
    if value.shape != ():
        raise ValueError(f"Checkpoint field {key!r} must be scalar, got {value.shape}")
    return value.item()


def checkpoint_payload(
    accumulator: RunningCovariance,
    next_index: int,
    dataset_path: Path,
    dataset_stat: os.stat_result,
    dataset_shape: tuple[int, ...],
    dataset_dtype: str,
    target_filters: int,
    chunk_size: int,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": np.asarray(CHECKPOINT_FORMAT_VERSION, dtype=np.int64),
        "preprocessing_id": np.asarray(PREPROCESSING_ID),
        "dataset_path": np.asarray(str(dataset_path.resolve())),
        "dataset_size_bytes": np.asarray(dataset_stat.st_size, dtype=np.int64),
        "dataset_mtime_ns": np.asarray(dataset_stat.st_mtime_ns, dtype=np.int64),
        "dataset_filters_shape": np.asarray(dataset_shape, dtype=np.int64),
        "dataset_filters_dtype": np.asarray(dataset_dtype),
        "target_filters": np.asarray(target_filters, dtype=np.int64),
        "chunk_size": np.asarray(chunk_size, dtype=np.int64),
        "next_index": np.asarray(next_index, dtype=np.int64),
        "n_samples": np.asarray(accumulator.n, dtype=np.int64),
        "mean_accumulator": accumulator.mean.astype(np.float64, copy=False),
        "m2_accumulator": accumulator.m2.astype(np.float64, copy=False),
        "saved_at_utc": np.asarray(datetime.now(timezone.utc).isoformat()),
    }


def save_checkpoint(
    path: Path,
    accumulator: RunningCovariance,
    next_index: int,
    dataset_path: Path,
    dataset_stat: os.stat_result,
    dataset_shape: tuple[int, ...],
    dataset_dtype: str,
    target_filters: int,
    chunk_size: int,
) -> None:
    atomic_savez(
        path,
        **checkpoint_payload(
            accumulator=accumulator,
            next_index=next_index,
            dataset_path=dataset_path,
            dataset_stat=dataset_stat,
            dataset_shape=dataset_shape,
            dataset_dtype=dataset_dtype,
            target_filters=target_filters,
            chunk_size=chunk_size,
        ),
    )
    print(
        f"[checkpoint] {path} at filter {next_index:,}/{target_filters:,}",
        flush=True,
    )


def load_checkpoint(
    path: Path,
    dataset_path: Path,
    dataset_stat: os.stat_result,
    dataset_shape: tuple[int, ...],
    dataset_dtype: str,
    target_filters: int,
    chunk_size: int,
) -> tuple[RunningCovariance, int]:
    required = {
        "checkpoint_format_version",
        "preprocessing_id",
        "dataset_path",
        "dataset_size_bytes",
        "dataset_mtime_ns",
        "dataset_filters_shape",
        "dataset_filters_dtype",
        "target_filters",
        "chunk_size",
        "next_index",
        "n_samples",
        "mean_accumulator",
        "m2_accumulator",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Checkpoint {path} is missing fields: {missing}")

        checks = [
            (
                int(scalar(data, "checkpoint_format_version")),
                CHECKPOINT_FORMAT_VERSION,
                "checkpoint format version",
            ),
            (str(scalar(data, "preprocessing_id")), PREPROCESSING_ID, "preprocessing"),
            (str(scalar(data, "dataset_path")), str(dataset_path.resolve()), "dataset path"),
            (int(scalar(data, "dataset_size_bytes")), dataset_stat.st_size, "dataset size"),
            (int(scalar(data, "dataset_mtime_ns")), dataset_stat.st_mtime_ns, "dataset mtime"),
            (str(scalar(data, "dataset_filters_dtype")), dataset_dtype, "dataset dtype"),
            (int(scalar(data, "target_filters")), target_filters, "target filter count"),
            (int(scalar(data, "chunk_size")), chunk_size, "chunk size"),
        ]
        for observed, expected, label in checks:
            if observed != expected:
                raise ValueError(
                    f"Cannot resume {path}: {label} is {observed!r}, expected {expected!r}"
                )
        observed_shape = tuple(int(v) for v in data["dataset_filters_shape"].tolist())
        if observed_shape != dataset_shape:
            raise ValueError(
                f"Cannot resume {path}: dataset shape is {observed_shape}, "
                f"expected {dataset_shape}"
            )

        next_index = int(scalar(data, "next_index"))
        n_samples = int(scalar(data, "n_samples"))
        mean = data["mean_accumulator"].astype(np.float64, copy=True)
        m2 = data["m2_accumulator"].astype(np.float64, copy=True)

    if next_index != n_samples:
        raise ValueError(
            f"Checkpoint invariant failed: next_index={next_index}, n_samples={n_samples}"
        )
    if not 0 <= next_index <= target_filters:
        raise ValueError(f"Checkpoint next_index {next_index} is outside the target range")
    if mean.shape != (9,) or m2.shape != (9, 9):
        raise ValueError(f"Checkpoint accumulator shapes are invalid: {mean.shape}, {m2.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(m2).all():
        raise ValueError("Checkpoint contains non-finite sufficient statistics")
    if not np.allclose(m2, m2.T, rtol=1e-12, atol=1e-6):
        raise ValueError("Checkpoint M2 matrix is not symmetric")

    print(f"[resume] Loaded {path} at filter {next_index:,}", flush=True)
    return RunningCovariance(n=n_samples, mean=mean, m2=m2), next_index


def paper_preprocess_chunk(filters_dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
    """Apply the notebook's float16 conversion and normalization exactly."""

    raw = filters_dataset[start:stop]
    values16 = raw.reshape(-1, 9).astype(np.float16)
    del raw
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.abs(values16).max(axis=1)
        denominator = np.where(denominator == 0, 1, denominator)[:, None]
        scaled16 = values16 / denominator
    del values16, denominator
    if not np.isfinite(scaled16).all():
        bad = int(scaled16.size - np.count_nonzero(np.isfinite(scaled16)))
        raise ValueError(
            f"Paper preprocessing produced {bad:,} non-finite values in filters "
            f"[{start:,}:{stop:,}]"
        )
    values64 = scaled16.astype(np.float64)
    del scaled16
    return values64


def deterministic_component_signs(components: np.ndarray) -> np.ndarray:
    """Canonicalize signs from loadings alone for stable signed axes.

    PCA signs are mathematically arbitrary. Scikit-learn 0.24 chooses signs from
    the left singular vectors, which would require another 99 GB data pass once
    the covariance eigenvectors are known. This equivalent V-based convention
    is deterministic from the saved basis itself and changes only axis signs.
    """

    result = components.copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    return result


def finalize_pca(accumulator: RunningCovariance) -> dict[str, np.ndarray]:
    if accumulator.n < 2:
        raise ValueError("At least two filters are required to compute sample covariance")
    covariance = accumulator.m2 / float(accumulator.n - 1)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order].T

    negative_tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-10
    if float(eigenvalues.min()) < -negative_tolerance:
        raise ValueError(f"Covariance has a materially negative eigenvalue: {eigenvalues.min()}")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    variance_total = float(eigenvalues.sum())
    if not np.isfinite(variance_total) or variance_total <= 0:
        raise ValueError(f"Invalid total explained variance: {variance_total}")

    components = deterministic_component_signs(components)
    explained_variance_ratio = eigenvalues / variance_total
    singular_values = np.sqrt(eigenvalues * float(accumulator.n - 1))
    orthonormality_error = float(
        np.max(np.abs(components @ components.T - np.eye(9, dtype=np.float64)))
    )
    if orthonormality_error > 1e-10:
        raise ValueError(f"Component orthonormality check failed: {orthonormality_error}")

    return {
        "components": components.astype(np.float64),
        "mean": accumulator.mean.astype(np.float64),
        "explained_variance": eigenvalues.astype(np.float64),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float64),
        "singular_values": singular_values.astype(np.float64),
        "covariance": covariance.astype(np.float64),
        "orthonormality_max_abs_error": np.asarray(orthonormality_error, dtype=np.float64),
    }


def artifact_payload(
    pca: dict[str, np.ndarray],
    accumulator: RunningCovariance,
    args: argparse.Namespace,
    dataset_path: Path,
    dataset_stat: os.stat_result,
    dataset_shape: tuple[int, ...],
    dataset_dtype: str,
    target_filters: int,
    full_filter_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    mean_error = pca["mean"] - PUBLISHED_FULL_DATASET_MEAN
    mean_max_abs_error = float(np.max(np.abs(mean_error)))
    validation_applicable = target_filters == full_filter_count
    validation_passed = mean_max_abs_error <= args.published_mean_tolerance
    command = " ".join(shlex.quote(value) for value in sys.argv)
    payload: dict[str, Any] = {
        # Existing analysis-consumer schema.
        "components": pca["components"],
        "mean": pca["mean"],
        "explained_variance": pca["explained_variance"],
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "singular_values": pca["singular_values"],
        # Reproducibility and validation provenance.
        "covariance": pca["covariance"],
        "artifact_format_version": np.asarray(ARTIFACT_FORMAT_VERSION, dtype=np.int64),
        "preprocessing_id": np.asarray(PREPROCESSING_ID),
        "preprocessing_description": np.asarray(
            "row-major reshape to N x 9; cast raw values to float16; divide each "
            "filter by its float16 maximum absolute value with zero rows unchanged; "
            "accumulate and eigendecompose sample covariance in float64"
        ),
        "fitting_method": np.asarray("streaming_merged_float64_covariance_eigh"),
        "component_sign_convention": np.asarray(
            "largest-absolute loading in each component is positive (canonical V-based sign)"
        ),
        "n_samples": np.asarray(accumulator.n, dtype=np.int64),
        "n_features": np.asarray(9, dtype=np.int64),
        "filter_index_start": np.asarray(0, dtype=np.int64),
        "filter_index_stop_exclusive": np.asarray(target_filters, dtype=np.int64),
        "full_dataset_filter_count": np.asarray(full_filter_count, dtype=np.int64),
        "is_full_dataset_fit": np.asarray(validation_applicable, dtype=np.bool_),
        "dataset_path": np.asarray(str(dataset_path.resolve())),
        "dataset_size_bytes": np.asarray(dataset_stat.st_size, dtype=np.int64),
        "dataset_mtime_ns": np.asarray(dataset_stat.st_mtime_ns, dtype=np.int64),
        "dataset_filters_shape": np.asarray(dataset_shape, dtype=np.int64),
        "dataset_filters_dtype": np.asarray(dataset_dtype),
        "source_dataset_doi": np.asarray("10.5281/zenodo.6371680"),
        "source_compressed_md5": np.asarray("11d0ca3f9c3f7b6e0f73d389db35105a"),
        "published_full_dataset_mean": PUBLISHED_FULL_DATASET_MEAN,
        "published_mean_signed_error": mean_error.astype(np.float64),
        "published_mean_max_abs_error": np.asarray(mean_max_abs_error, dtype=np.float64),
        "published_mean_tolerance": np.asarray(
            args.published_mean_tolerance, dtype=np.float64
        ),
        "published_mean_validation_applicable": np.asarray(
            validation_applicable, dtype=np.bool_
        ),
        "published_mean_validation_passed": np.asarray(validation_passed, dtype=np.bool_),
        "orthonormality_max_abs_error": pca["orthonormality_max_abs_error"],
        "explained_variance_ratio_sum": np.asarray(
            float(pca["explained_variance_ratio"].sum()), dtype=np.float64
        ),
        "chunk_size": np.asarray(args.chunk_size, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "created_at_utc": np.asarray(datetime.now(timezone.utc).isoformat()),
        "hostname": np.asarray(platform.node()),
        "python_version": np.asarray(platform.python_version()),
        "numpy_version": np.asarray(np.__version__),
        "h5py_version": np.asarray(h5py.__version__),
        "command": np.asarray(command),
    }
    return payload


def print_progress(
    next_index: int,
    target_filters: int,
    run_start_index: int,
    run_start_time: float,
) -> None:
    elapsed = max(time.monotonic() - run_start_time, 1e-9)
    processed_this_run = next_index - run_start_index
    rate = processed_this_run / elapsed
    remaining = target_filters - next_index
    eta_seconds = remaining / rate if rate > 0 else float("inf")
    percent = 100.0 * next_index / max(target_filters, 1)
    print(
        f"[progress] {next_index:,}/{target_filters:,} ({percent:.2f}%) | "
        f"{rate:,.0f} filters/s | elapsed {elapsed / 60.0:.1f} min | "
        f"ETA {eta_seconds / 3600.0:.2f} h",
        flush=True,
    )


def default_checkpoint_path(output_path: Path) -> Path:
    if output_path.suffix == ".npz":
        return output_path.with_name(f"{output_path.stem}.checkpoint.npz")
    return output_path.with_name(f"{output_path.name}.checkpoint.npz")


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.checkpoint_every_chunks <= 0:
        raise ValueError("--checkpoint-every-chunks must be positive")
    if args.max_filters < 0:
        raise ValueError("--max-filters must be non-negative")
    if args.max_chunks_per_run < 0:
        raise ValueError("--max-chunks-per-run must be non-negative")
    if args.published_mean_tolerance < 0:
        raise ValueError("--published-mean-tolerance must be non-negative")

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(output_path)
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_path == checkpoint_path:
        raise ValueError("--output and --checkpoint must be different files")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Final artifact already exists: {output_path}. Use --force to recompute it."
        )
    if checkpoint_path.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"Checkpoint exists but --no-resume was selected: {checkpoint_path}"
        )
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[force] Removed stale checkpoint {checkpoint_path}", flush=True)

    install_signal_handlers()
    dataset_stat = dataset_path.stat()
    overall_start_time = time.monotonic()
    print(f"[start] dataset={dataset_path.resolve()}", flush=True)
    print(f"[start] output={output_path}", flush=True)
    print(f"[start] checkpoint={checkpoint_path}", flush=True)
    print(f"[start] chunk_size={args.chunk_size:,}", flush=True)

    with h5py.File(dataset_path, "r") as handle:
        if "filters" not in handle:
            raise KeyError(f"{dataset_path} does not contain /filters")
        filters_dataset = handle["filters"]
        dataset_shape = tuple(int(v) for v in filters_dataset.shape)
        dataset_dtype = str(filters_dataset.dtype)
        if len(dataset_shape) != 3 or dataset_shape[1:] != (3, 3):
            raise ValueError(f"Expected /filters shape (N, 3, 3), got {dataset_shape}")
        full_filter_count = dataset_shape[0]
        target_filters = (
            full_filter_count
            if args.max_filters == 0
            else min(args.max_filters, full_filter_count)
        )
        if target_filters < 2:
            raise ValueError(f"Target filter count must be at least two, got {target_filters}")
        if args.max_filters > full_filter_count:
            print(
                f"[start] Requested {args.max_filters:,} filters; clamped to "
                f"dataset size {full_filter_count:,}.",
                flush=True,
            )
        print(
            f"[start] /filters shape={dataset_shape}, dtype={dataset_dtype}, "
            f"target={target_filters:,}",
            flush=True,
        )

        if checkpoint_path.exists() and args.resume and not args.force:
            accumulator, next_index = load_checkpoint(
                checkpoint_path,
                dataset_path,
                dataset_stat,
                dataset_shape,
                dataset_dtype,
                target_filters,
                args.chunk_size,
            )
        else:
            accumulator = RunningCovariance.empty()
            next_index = 0

        run_start_index = next_index
        run_start_time = time.monotonic()
        chunks_this_run = 0
        chunks_since_checkpoint = 0

        while next_index < target_filters:
            chunk_start = next_index
            stop = min(chunk_start + args.chunk_size, target_filters)
            chunk_start_time = time.monotonic()
            values64 = paper_preprocess_chunk(filters_dataset, chunk_start, stop)
            accumulator.update(values64)
            del values64
            next_index = stop
            chunks_this_run += 1
            chunks_since_checkpoint += 1
            chunk_seconds = time.monotonic() - chunk_start_time
            print(
                f"[chunk] completed [{chunk_start:,}:{next_index:,}] in "
                f"{chunk_seconds:.2f}s",
                flush=True,
            )
            print_progress(next_index, target_filters, run_start_index, run_start_time)

            should_checkpoint = (
                chunks_since_checkpoint >= args.checkpoint_every_chunks
                or next_index == target_filters
                or _STOP_SIGNAL is not None
                or (
                    args.max_chunks_per_run > 0
                    and chunks_this_run >= args.max_chunks_per_run
                )
            )
            if should_checkpoint:
                save_checkpoint(
                    checkpoint_path,
                    accumulator,
                    next_index,
                    dataset_path,
                    dataset_stat,
                    dataset_shape,
                    dataset_dtype,
                    target_filters,
                    args.chunk_size,
                )
                chunks_since_checkpoint = 0

            if _STOP_SIGNAL is not None and next_index < target_filters:
                print("[paused] Signal checkpoint complete; rerun to resume.", flush=True)
                return 75
            if (
                args.max_chunks_per_run > 0
                and chunks_this_run >= args.max_chunks_per_run
                and next_index < target_filters
            ):
                print("[paused] Chunk limit reached; rerun to resume.", flush=True)
                return 75

    if accumulator.n != target_filters or next_index != target_filters:
        raise RuntimeError(
            f"Completion invariant failed: n={accumulator.n}, next={next_index}, "
            f"target={target_filters}"
        )

    pca = finalize_pca(accumulator)
    elapsed_seconds = time.monotonic() - overall_start_time
    payload = artifact_payload(
        pca=pca,
        accumulator=accumulator,
        args=args,
        dataset_path=dataset_path,
        dataset_stat=dataset_stat,
        dataset_shape=dataset_shape,
        dataset_dtype=dataset_dtype,
        target_filters=target_filters,
        full_filter_count=full_filter_count,
        elapsed_seconds=elapsed_seconds,
    )
    mean_error = float(payload["published_mean_max_abs_error"])
    validation_applicable = bool(payload["published_mean_validation_applicable"])
    validation_passed = bool(payload["published_mean_validation_passed"])
    print(f"[validate] computed mean: {pca['mean'].tolist()}", flush=True)
    print(
        f"[validate] paper-mean max absolute error: {mean_error:.9g} "
        f"(tolerance {args.published_mean_tolerance:.9g}; "
        f"applicable={validation_applicable})",
        flush=True,
    )
    print(
        "[validate] explained variance ratio: "
        f"{pca['explained_variance_ratio'].tolist()}",
        flush=True,
    )
    print(
        f"[validate] EVR sum={float(pca['explained_variance_ratio'].sum()):.12f}, "
        f"orthonormality error={float(pca['orthonormality_max_abs_error']):.3e}",
        flush=True,
    )
    if (
        validation_applicable
        and not validation_passed
        and not args.allow_published_mean_mismatch
    ):
        raise ValueError(
            "Full-data mean does not match the published CNN Filter DB centering "
            "vector. The final artifact was not written; inspect the retained checkpoint "
            "or use --allow-published-mean-mismatch only after explaining the discrepancy."
        )

    atomic_savez(output_path, **payload)
    print(f"[done] Wrote PCA artifact: {output_path}", flush=True)
    print(f"[done] Samples: {accumulator.n:,}; elapsed: {elapsed_seconds / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
