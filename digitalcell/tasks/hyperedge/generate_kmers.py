from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
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


# Hyperedges per worker task; modest so per-worker k-mer memory + in-flight pickle stay small.
CHUNK_CONCATEMERS = 200_000
# Consolidate running totals once buffered unique rows exceed this (bounds parent memory).
MERGE_ROW_BUDGET = 150_000_000


def count_kmers_chunk(size: int, chunk: np.ndarray):
    """Enumerate size-mers for one chunk; return per-chunk (unique_kmers, counts) as int32/int64.

    Hyperedges are sorted (read_clusters), so combinations() yields sorted canonical k-mers.
    Per-chunk uniquing keeps the worker result (and the parent accumulator) small.
    """
    rows = []
    for hyperedge in chunk:
        if len(hyperedge) >= size:
            rows.extend(combinations(hyperedge, size))
    if not rows:
        return np.empty((0, size), dtype=np.int32), np.empty((0,), dtype=np.int64)
    keys, counts = np.unique(np.asarray(rows, dtype=np.int32), axis=0, return_counts=True)
    return keys, counts.astype(np.int64)


def build_kmers_parallel(size: int, new_data: np.ndarray, min_freq_cutoff: int, max_workers: int):
    """Count size-mer frequencies, keeping those with count >= min_freq_cutoff.

    Parallel but memory-safe. Forked workers each uniquely-count one modest chunk and return small
    (keys, counts) arrays; the parent merges them into running int32/int64 arrays (~6x lighter than a
    Python Counter). Only a bounded WINDOW of tasks is in flight at once, so we never pickle the whole
    edge list into the pool queue (the bug that OOM'd a submit-all-at-once version), and the run
    completes without leaving orphaned workers. Result is identical (row order aside) to a global Counter.
    """
    n = len(new_data)
    if n == 0:
        return np.empty((0, size), dtype=np.int32), np.empty((0,), dtype=np.int64)

    n_chunks = max(max_workers, math.ceil(n / CHUNK_CONCATEMERS))
    chunks = [ch for ch in np.array_split(new_data, n_chunks) if len(ch)]

    run_keys = np.empty((0, size), dtype=np.int32)
    run_counts = np.empty((0,), dtype=np.int64)
    pending_keys: List[np.ndarray] = []
    pending_counts: List[np.ndarray] = []
    pending_rows = 0

    def consolidate():
        nonlocal run_keys, run_counts, pending_keys, pending_counts, pending_rows
        if not pending_keys:
            return
        all_keys = np.concatenate([run_keys, *pending_keys], axis=0)
        all_counts = np.concatenate([run_counts, *pending_counts])
        merged_keys, inv = np.unique(all_keys, axis=0, return_inverse=True)
        merged_counts = np.zeros(merged_keys.shape[0], dtype=np.int64)
        np.add.at(merged_counts, inv.ravel(), all_counts)
        run_keys, run_counts = merged_keys, merged_counts
        pending_keys, pending_counts, pending_rows = [], [], 0

    max_inflight = max(2, max_workers * 2)
    chunk_iter = iter(chunks)
    inflight = set()
    # spawn (not fork): workers start fresh and receive their chunk by pickle, so they do NOT
    # COW-inherit the parent's multi-GB edge list (which, with fork, balloons to ~Nworkers x edge_list
    # and OOMs). Requires this module to be importable with an `if __name__ == "__main__"` guard,
    # i.e. run via `python generate_kmers.py ...`, never `python -c`.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
        for _ in range(max_inflight):
            ch = next(chunk_iter, None)
            if ch is None:
                break
            inflight.add(pool.submit(count_kmers_chunk, size, ch))
        with tqdm(total=len(chunks), desc=f"k={size}") as pbar:
            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    keys, counts = fut.result()
                    if keys.shape[0]:
                        pending_keys.append(keys)
                        pending_counts.append(counts)
                        pending_rows += int(keys.shape[0])
                        if pending_rows >= MERGE_ROW_BUDGET:
                            consolidate()
                    pbar.update(1)
                    ch = next(chunk_iter, None)
                    if ch is not None:
                        inflight.add(pool.submit(count_kmers_chunk, size, ch))
    consolidate()

    mask = run_counts >= min_freq_cutoff
    return run_keys[mask], run_counts[mask]


# ----------------------------------------------------------------------------------------
# External (disk-partitioned) counter -- for inputs too large to hold a global unique in RAM.
#
# Some datasets (e.g. GM12878: 78M concatemers, size up to 25) produce >10^10 k-mers with
# multiplicity (k=5 ~= 25 billion), far more than any in-memory unique can hold. We instead
# hash-partition every k-mer to disk so identical k-mers always land in the same partition,
# then count each partition independently (each fits in RAM). Memory is bounded (~scatter
# buffer per worker + one partition per counter); the result is identical (order aside) to
# build_kmers_parallel. Node ids are < 2^15, so k-mers are stored as int16 to halve disk + IO.
# ----------------------------------------------------------------------------------------

N_PARTITIONS = 512                 # hash buckets; partition rows ~= total_kmers / N_PARTITIONS
SCATTER_ROW_CHUNK = 8_000_000      # enumerate this many k-mers before binning (bounds peak rows)
SCATTER_FLUSH_ROWS = 16_000_000    # flush per-partition buffers to disk past this many buffered rows
COUNT_CONCURRENCY = 8              # partitions counted at once (each ~ total/P rows -> bounded RAM)
EXTERNAL_THRESHOLD = 1_500_000_000 # use the external counter when est. k-mers exceed this
INT16_MAX_NODES = 32768            # int16 node-id ceiling (node_num must be below this)


def _partition_ids(arr: np.ndarray, size: int, n_parts: int) -> np.ndarray:
    """Deterministic hash of each (sorted) k-mer row to a partition in [0, n_parts).
    Identical rows -> identical partition (required for correctness)."""
    h = np.zeros(arr.shape[0], dtype=np.uint64)
    for c in range(size):
        h = h * np.uint64(1000003) + arr[:, c].astype(np.uint64)
    return (h % np.uint64(n_parts)).astype(np.int64)


def scatter_worker(args) -> int:
    """Phase 1: read one shard of new_data, enumerate its k-mers, and append them (as int16)
    to per-partition files for this worker: <tmp>/k{size}_p{p}_w{wid}.i16."""
    shard_path, size, n_parts, tmpdir, wid = args
    shard = np.load(shard_path, allow_pickle=True)
    bufs: List[List[np.ndarray]] = [[] for _ in range(n_parts)]
    buffered = 0

    def emit(rows: list) -> None:
        nonlocal buffered
        arr = np.asarray(rows, dtype=np.int16)          # (m, size); rows are sorted k-tuples
        part = _partition_ids(arr, size, n_parts)
        order = np.argsort(part, kind="stable")
        arr = arr[order]
        bounds = np.searchsorted(part[order], np.arange(n_parts + 1))
        for p in range(n_parts):
            s, e = bounds[p], bounds[p + 1]
            if e > s:
                bufs[p].append(arr[s:e])
        buffered += arr.shape[0]

    def flush() -> None:
        nonlocal buffered
        for p in range(n_parts):
            if bufs[p]:
                blk = np.concatenate(bufs[p], axis=0)
                with open(os.path.join(tmpdir, f"k{size}_p{p}_w{wid}.i16"), "ab") as fh:
                    blk.tofile(fh)
                bufs[p] = []
        buffered = 0

    rows: list = []
    for he in shard:
        if len(he) >= size:
            rows.extend(combinations(he, size))
            if len(rows) >= SCATTER_ROW_CHUNK:
                emit(rows); rows = []
                if buffered >= SCATTER_FLUSH_ROWS:
                    flush()
    if rows:
        emit(rows)
    flush()
    return wid


def count_partition_worker(args):
    """Phase 2: load all workers' rows for one partition, unique+count, keep count >= cutoff."""
    size, p, n_workers, tmpdir, min_freq_cutoff = args
    parts = []
    for w in range(n_workers):
        path = os.path.join(tmpdir, f"k{size}_p{p}_w{w}.i16")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            parts.append(np.fromfile(path, dtype=np.int16).reshape(-1, size))
    if not parts:
        return None
    allrows = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    keys, counts = np.unique(allrows, axis=0, return_counts=True)
    mask = counts >= min_freq_cutoff
    if not mask.any():
        return None
    return keys[mask].astype(np.int32), counts[mask].astype(np.int64)


def build_kmers_external(size: int, new_data: np.ndarray, min_freq_cutoff: int,
                         max_workers: int, scratch_root: Path, node_num: int):
    """Memory-bounded, disk-partitioned k-mer counter. Identical output to build_kmers_parallel."""
    n = len(new_data)
    if n == 0:
        return np.empty((0, size), dtype=np.int32), np.empty((0,), dtype=np.int64)
    if node_num >= INT16_MAX_NODES:
        raise ValueError(f"node_num {node_num} >= {INT16_MAX_NODES}; int16 k-mer storage unsafe.")

    tmpdir = Path(scratch_root) / f".kmer_scratch_k{size}"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    ctx = mp.get_context("spawn")
    try:
        # write shards of new_data to disk so workers load them (spawn) instead of inheriting RAM
        shard_paths = []
        for wid, shard in enumerate(np.array_split(new_data, max_workers)):
            sp = tmpdir / f"shard_{wid}.npy"
            np.save(sp, shard, allow_pickle=True)
            shard_paths.append(str(sp))

        # Phase 1: scatter k-mers to per-(partition, worker) files
        scatter_args = [(shard_paths[w], size, N_PARTITIONS, str(tmpdir), w) for w in range(len(shard_paths))]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            list(tqdm(pool.map(scatter_worker, scatter_args), total=len(scatter_args), desc=f"k{size} scatter"))
        for sp in shard_paths:
            os.remove(sp)

        # Phase 2: count each partition independently (bounded concurrency -> bounded RAM)
        count_args = [(size, p, len(shard_paths), str(tmpdir), min_freq_cutoff) for p in range(N_PARTITIONS)]
        results = []
        with ProcessPoolExecutor(max_workers=min(COUNT_CONCURRENCY, max_workers), mp_context=ctx) as pool:
            for r in tqdm(pool.map(count_partition_worker, count_args), total=N_PARTITIONS, desc=f"k{size} count"):
                if r is not None:
                    results.append(r)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not results:
        return np.empty((0, size), dtype=np.int32), np.empty((0,), dtype=np.int64)
    keys = np.concatenate([r[0] for r in results], axis=0)
    counts = np.concatenate([r[1] for r in results])
    order = np.lexsort(keys.T[::-1])          # deterministic (lexicographic) order
    return keys[order], counts[order]


def estimate_total_kmers(new_data: np.ndarray, size: int) -> int:
    """Number of k-mers (with multiplicity) new_data would yield = sum_i C(len_i, size)."""
    sizes = np.fromiter((len(e) for e in new_data), dtype=np.int64, count=len(new_data))
    s = int(sizes.max()) if sizes.size else 0
    table = np.array([math.comb(v, size) if v >= size else 0 for v in range(s + 1)], dtype=np.float64)
    return int(table[sizes].sum()) if sizes.size else 0


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

        total_est = estimate_total_kmers(new_data, size)
        if total_est > EXTERNAL_THRESHOLD:
            print(f"k={size}: ~{total_est:,} k-mers -> external (disk-partitioned) counter")
            keys, freqs = build_kmers_external(
                size=size,
                new_data=new_data,
                min_freq_cutoff=cfg.min_freq_cutoff,
                max_workers=max_workers,
                scratch_root=cfg.temp_dir,
                node_num=node_num,
            )
        else:
            print(f"k={size}: ~{total_est:,} k-mers -> in-memory counter")
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
    import argparse

    ap = argparse.ArgumentParser(
        description="Build k-mer hyperedges + frequency counts from edge_list.npy in --temp_dir.")
    ap.add_argument("--temp_dir", required=True,
                    help="dir containing edge_list.npy; all_{k}_counter.npy / all_{k}_freq_counter.npy are written here")
    ap.add_argument("--max_cluster_size", type=int, default=25)
    ap.add_argument("--k_list", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--min_freq_cutoff", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=100000)
    args = ap.parse_args()

    main(Config(
        max_cluster_size=args.max_cluster_size,
        k_list=args.k_list,
        temp_dir=Path(args.temp_dir),
        min_freq_cutoff=args.min_freq_cutoff,
        resolution=args.resolution,
    ))
