from __future__ import annotations

import math
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

from digitalcell.data import constants

"""
This file is adapted from `generate_kmers.py` in the MATCHA repository
The original file can be found at: https://github.com/ma-compbio/MATCHA/blob/master/Code/generate_kmers.py

The main script builds a list of k-mer hyperedges without duplicates. It also saves the frequency counts for each k-mer hyperedge.
"""

def get_available_cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK",
           os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count())))

@dataclass(frozen=True)
class Config:
    max_cluster_size: int
    k_list: List[int]
    temp_dir: Path
    min_freq_cutoff: int
    resolution: int


def filter_data_by_size(
    data: Iterable[np.ndarray], 
    size: int, 
    max_size: int
) -> np.ndarray:
    """Keep rows with length in [size, max_size]."""
    
    kept = [np.asarray(edge, dtype=int) for edge in data if size <= len(edge) <= max_size]
    return np.asarray(kept, dtype=object)  # ragged rows -> object dtype


def count_kmers_chunk(size: int, chunk: np.ndarray) -> Counter:
    c = Counter()
    for hyperedge in chunk:
        if len(hyperedge) < size:
            continue

        c.update(combinations(hyperedge, size))
    return c

def build_kmers_parallel(size: int, new_data: np.ndarray, min_freq_cutoff: int, max_workers: int):
    chunks = np.array_split(new_data, max_workers * 4)  # a few chunks per worker

    total = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(count_kmers_chunk, size, ch) for ch in chunks if len(ch)]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"k={size} | merging chunks"
        ):
            total.update(fut.result())

    items = [(k, v) for k, v in total.items() if v >= min_freq_cutoff]
    if not items:
        return np.empty((0, size), dtype=int), np.empty((0,), dtype=int)

    ks, fs = zip(*items)
    keys  = np.array(ks, dtype=int)   # (N, size)
    freqs = np.array(fs, dtype=int)   # (N,)
    return keys, freqs


def summarize(freqs: np.ndarray) -> None:
    print("Quick summarize")
    print("total data", len(freqs))
    for c in range(2, 9):
        print(f">= {c}", int((freqs >= c).sum()))


def main(cfg: Config) -> None:
    node_num = constants.get_chrom_offset(resolution=cfg.resolution)[-1]
    print(f"Total number of nodes: {node_num}")

    data = np.load(cfg.temp_dir / "edge_list.npy", allow_pickle=True)

    max_workers = max(1, min(get_available_cpus(), node_num))
    print(f"Found {max_workers} available CPU cores for parallel processing.")

    for size in cfg.k_list:
        new_data = filter_data_by_size(data, size=size, max_size=cfg.max_cluster_size)

        keys, freqs = build_kmers_parallel(
            size=size,
            new_data=new_data,
            min_freq_cutoff=cfg.min_freq_cutoff,
            max_workers=max_workers,
        )

        if keys.size:
            print(f"\nCollected {keys.shape[0]} keys for k={size} (shape={keys.shape})")
            np.save(cfg.temp_dir / f"all_{size}_counter.npy", keys)
            np.save(cfg.temp_dir / f"all_{size}_freq_counter.npy", freqs)
            summarize(freqs)
        else:
            print(f"\nNo keys collected for k={size}.")



if __name__ == "__main__":
    cfg = Config(
        max_cluster_size=25,
        k_list=[3, 4, 5],
        temp_dir=Path("/path/to/directory/containing/edge_list.npy"),
        min_freq_cutoff=2,
        resolution=100000,
    )
    main(cfg)
