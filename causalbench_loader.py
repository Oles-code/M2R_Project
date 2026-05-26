"""
causalbench_loader
==================
A small wrapper around the causalscbench data pipeline that fixes two
bugs in the upstream package (v1.1.2):

  1. The bundled downloader (causalscbench/data_access/utils/download.py)
     calls `gdown.download(url, ...)` against Figshare URLs. gdown only
     handles Google Drive, so it silently fails and leaves 0-byte stubs.

  2. The bundled URLs point at `plus.figshare.com`, which sits behind an
     AWS-WAF challenge that blocks automated requests.

This module replaces the download step with a resumable, size-validated
HTTP download against `ndownloader.figshare.com` (the working host), then
delegates to causalscbench's own preprocessing + splitting logic for
everything downstream.

Public API:

    download_raw_data(data_dir)
        Fetch k562.h5ad (~10.6 GB) and rpe1.h5ad (~8.7 GB) into data_dir.
        Resumable: if the file already exists with a smaller size, it
        will request only the missing tail via an HTTP Range header.

    preprocess(data_dir, dataset_name)
        Run causalscbench's normalize + log1p + rarely-perturbed-gene
        filter on one dataset. Returns the path to the cached .npz.

    load_split(npz_path, regime, subset_data=1.0, partial_fraction=0.5)
        Stratified 80/20 train/test split (the test split is held out by
        causalscbench) and apply the requested intervention regime.
        Returns (expression_matrix, interventions, gene_names).
"""

import http.client
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import List, Tuple

import numpy as np


# Figshare file IDs (same as causalscbench but rerouted off plus.*) +
# expected sizes from the Figshare API. The size check is what protects
# against the OSError("truncated file") you get when a partial download
# is fed to h5py.
_RAW_FILES = {
    "k562.h5ad": {
        "url": "https://ndownloader.figshare.com/files/35773219",
        "size": 10_661_879_995,
    },
    "rpe1.h5ad": {
        "url": "https://ndownloader.figshare.com/files/35775606",
        "size": 8_700_873_216,
    },
}

# Plain browser UA is enough to satisfy Figshare's basic checks.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# Figshare's S3 CDN cuts long downloads roughly every 1 GB. The loader
# handles this transparently by catching connection drops and re-issuing
# a Range request for the remaining bytes, up to _MAX_ATTEMPTS times.
_MAX_ATTEMPTS = 100
_BACKOFF_S = 5

# Network errors that should be treated as "connection dropped, resume":
_TRANSIENT = (
    urllib.error.URLError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionResetError,
    TimeoutError,
    socket.timeout,
    OSError,
)


def _download_one_attempt(url: str, dest: str, expected_size: int) -> int:
    """Single attempt. Returns the number of bytes on disk after returning.

    Raises a _TRANSIENT exception if the connection drops mid-transfer.
    Raises RuntimeError if the server returns a bad status / size mismatch
    that won't be fixed by retrying.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == expected_size:
        return have
    if have > expected_size:
        print(f"  ⚠ {os.path.basename(dest)} larger than expected; restarting from scratch")
        os.remove(dest)
        have = 0

    headers = {"User-Agent": _UA}
    mode = "wb"
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"  resuming {os.path.basename(dest)} at {have/1e9:.2f} / {expected_size/1e9:.2f} GB")
    else:
        print(f"  downloading {os.path.basename(dest)} ({expected_size/1e9:.2f} GB) from {url}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
        got = have
        while True:
            chunk = r.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            sys.stdout.write(
                f"\r    {got/1e9:6.2f} / {expected_size/1e9:6.2f} GB"
                f"  ({100*got/expected_size:5.1f}%)"
            )
            sys.stdout.flush()
        sys.stdout.write("\n")
    return os.path.getsize(dest)


def _download_one(url: str, dest: str, expected_size: int) -> None:
    """Download `url` to `dest` with automatic resume on connection drops."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == expected_size:
        gb = expected_size / 1e9
        print(f"  ✓ {os.path.basename(dest)} already complete ({gb:.2f} GB)")
        return

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            got = _download_one_attempt(url, dest, expected_size)
        except _TRANSIENT as e:
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            sys.stdout.write("\n")
            print(f"  ⟳ attempt {attempt}/{_MAX_ATTEMPTS} dropped at "
                  f"{got/1e9:.2f} GB ({type(e).__name__}: {e}); retrying in {_BACKOFF_S}s")
            time.sleep(_BACKOFF_S)
            continue

        if got == expected_size:
            return
        # Server closed the stream cleanly but we got fewer bytes than
        # expected (this is the case that hit you on K562 at 1.0 GB).
        sys.stdout.write("\n")
        print(f"  ⟳ attempt {attempt}/{_MAX_ATTEMPTS} ended early at "
              f"{got/1e9:.2f} / {expected_size/1e9:.2f} GB; resuming")
        time.sleep(1)

    raise RuntimeError(
        f"{dest}: still short ({os.path.getsize(dest)} of {expected_size} bytes) "
        f"after {_MAX_ATTEMPTS} attempts. Bump _MAX_ATTEMPTS or check your network."
    )


def download_raw_data(data_dir: str, files: List[str] = None) -> None:
    """Download the listed raw .h5ad files into `data_dir`.

    Args:
        data_dir: target directory; created if missing.
        files:    subset of `_RAW_FILES.keys()`. Defaults to all.
    """
    os.makedirs(data_dir, exist_ok=True)
    for fname in files or list(_RAW_FILES):
        meta = _RAW_FILES[fname]
        _download_one(meta["url"], os.path.join(data_dir, fname), meta["size"])


_DATASET_TO_H5AD = {
    "weissmann_k562": ("k562.h5ad", "dataset_k562"),
    "weissmann_rpe1": ("rpe1.h5ad", "dataset_rpe1"),
}


def preprocess(data_dir: str, dataset_name: str) -> str:
    """Run causalscbench preprocessing for one dataset.

    Loads the raw .h5ad with scanpy, normalizes counts per cell, applies
    log1p, drops genes whose perturbation has < 100 cells, and writes a
    cached .npz containing (expression_matrix, var_names, interventions).
    Returns the path to that .npz.
    """
    from causalscbench.data_access.create_dataset import CreateDataset

    if dataset_name not in _DATASET_TO_H5AD:
        raise ValueError(f"Unknown dataset {dataset_name!r}; expected one of {list(_DATASET_TO_H5AD)}")
    h5ad_name, npz_basename = _DATASET_TO_H5AD[dataset_name]

    h5ad_path = os.path.join(data_dir, h5ad_name)
    expected = _RAW_FILES[h5ad_name]["size"]
    actual = os.path.getsize(h5ad_path) if os.path.exists(h5ad_path) else 0
    if actual != expected:
        raise FileNotFoundError(
            f"{h5ad_path} is missing or truncated ({actual} of {expected} bytes). "
            f"Run download_raw_data({data_dir!r}) first."
        )

    # filter=False skips the (also gated) summary-stats Excel download.
    creator = CreateDataset(data_directory=data_dir, filter=False)
    return creator.preprocess_and_save(h5ad_path, None, npz_basename)


def load_split(
    npz_path: str,
    regime: str = "observational",
    subset_data: float = 1.0,
    partial_fraction: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Apply the regime filter and return (expression_matrix, interventions, gene_names).

    `regime` is one of:
      - "observational"             control cells only
      - "partial_interventional"    control + interventions on `partial_fraction`
                                    of the perturbed genes (default 0.5)
      - "full_interventional"       control + interventions on all perturbed genes
    """
    from causalscbench.data_access.utils.splitting import DatasetSplitter

    splitter = DatasetSplitter(npz_path, subset_data=subset_data)

    if regime == "observational":
        expr, interventions, gene_names = splitter.get_observational()
    elif regime == "partial_interventional":
        expr, interventions, gene_names = splitter.get_partial_interventional(
            fraction=partial_fraction, seed=seed
        )
    elif regime == "full_interventional":
        expr, interventions, gene_names = splitter.get_interventional()
    else:
        raise ValueError(f"Unknown regime {regime!r}")

    # The (partial|full) interventional paths return interventions as an
    # itertools.compress iterator — consumers expect a list.
    return expr, list(interventions), gene_names
