import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import scipy
import torch

import cooler
from digitalcell.data import constants
from digitalcell.data.toeplitz_normalize import toeplitz_normalize

def basepair_to_bin(
    chromosome: int, 
    base_pair: int,
    chrom_offset: torch.Tensor,
    resolution: int
) -> int:
    """
    Converts a genomic locus (chromosome and base pair position) to a bin index based on a fixed resolution.

    Parameters:
    ----------
    chromosome : int
        Chromosome number (1-22 for autosomes, 23 for X, 24 for Y).
    base_pair : int
        Base pair position on the chromosome.
    Returns:
    -------
    int
        Bin index corresponding to the genomic locus.
    """
    
    if chromosome < 1 or chromosome > 22:
        raise ValueError("Chromosome must be between 1 and 22.")
    if base_pair < 0:
        raise ValueError("Base pair position must be non-negative.")

    offset = chrom_offset[chromosome - 1] # this is resolution-dependent
    midpoint = math.floor(base_pair / resolution) # works at any resolution
    
    # make sure index does not fall off chromosome
    if midpoint + offset > chrom_offset[chromosome] - 1:
        return None
    else:
        bin_idx = midpoint + offset
        return bin_idx.item()
    # bin_idx = min(midpoint + offset, chrom_offset[chromosome].item() - 1)

    # maybe needs to be tensor for indexing. Keep an eye on this.
    # return bin_idx.item()

def read_clusters(
    path: str,
    resolution: int
) -> list:
    """
    Reads cluster information from a file.

    Parameters:
    ----------
    path : str
        Path to the file containing cluster data.  

    Returns:
    -------
    clusters : list
        Nested list of clusters. Outer list is the hyperedges and inner list is the bin indices of each node in the hyperedge for the given resolution.
    """

    chrom_sizes = constants.get_chrom_sizes()
    chrom_bins = torch.ceil(chrom_sizes / resolution).int()
    chrom_offset = torch.cat((torch.zeros(1, dtype=torch.int32), chrom_bins.cumsum(dim=0, dtype=torch.int32)))
    # Pure-Python int offsets so per-locus binning avoids torch tensor indexing -- orders of
    # magnitude faster on large cluster files, and produces bins identical to basepair_to_bin().
    # off[c-1] = first bin of chromosome c; off[c] = first bin of chromosome c+1 (exclusive end).
    off = chrom_offset.tolist()

    clusters = []
    with Path(path).open() as f:
        for line in f:
            parts = line.split()                 # splits on tabs/whitespace; parts[0] is the cluster id
            if not parts:
                continue
            loci = set()
            for t in parts[1:]:
                chrom_tok, sep, pos_tok = t.partition(':')
                if not sep or chrom_tok[:3] != 'chr':
                    continue
                chrom_num = chrom_tok[3:]
                if not (chrom_num.isdigit() and pos_tok.isdigit()):
                    continue
                chrom = int(chrom_num)
                if chrom < 1 or chrom > 22:       # autosomes only (matches the original ^chr([0-9]+):(\d+)$ regex)
                    continue
                bin_idx = int(pos_tok) // resolution + off[chrom - 1]
                if bin_idx < off[chrom]:          # drop loci that fall off the chromosome end (== basepair_to_bin None)
                    loci.add(bin_idx)
            clusters.append(sorted(loci))         # sorted hyperedge of bin ids (required by generate_kmers)
    return clusters

def build_reference_coordinates(
    resolution: int
) -> tuple[int, torch.Tensor, torch.Tensor]:
    
    chrom_sizes = constants.get_chrom_sizes()
    chrom_lengths = torch.ceil(chrom_sizes / resolution).int() # length in bins
    chromosomes = torch.repeat_interleave(torch.arange(constants.NUM_CHROM, dtype=torch.int32), chrom_lengths)
    
    num_bins = int(chrom_lengths.sum().item())
    start_coords = resolution * torch.cat([torch.arange(0, chrom_lengths[chrom], dtype=torch.int32) for chrom in range(constants.NUM_CHROM)])
    end_coords = start_coords + resolution

    return num_bins, chromosomes, start_coords, end_coords

def build_cooler_file(
    clusters: Iterable,
    resolution: int,
    build_mcool_file: bool = False,
    save_dir: str = ""
) -> None:
    
    """
    Resolution should be same resolution as clusters list
    """
    
    """
    Converts list of hyperedges to contact map of ``virtual pairs,'' i.e., clique-expansion.
    """

    num_bins, chromosomes, start_coords, end_coords = build_reference_coordinates(resolution)

    # Memory-bounded clique expansion. A naive (rows, cols) Python-list accumulation holds every
    # pair with multiplicity (~1.6B pairs for the 78M-concatemer GM12878 set at 5kb -> ~75GB of
    # Python ints -> OOM). Instead: bucket concatemers by length, vectorize each length's
    # upper-triangle pairs (np.triu_indices), and encode each (row, col) as one int64 key
    # (row*num_bins + col). A single np.unique gives the deduplicated, *sorted* (row, col, count) --
    # sorted by key == sorted by (bin1, bin2), i.e. exactly cooler's pixel order. We free the
    # multi-GB `clusters` list before building the cooler and stream the sorted pixels to
    # create_cooler in chunks (no giant CSV dump, no lexsort), keeping peak memory bounded.
    nb = np.int64(num_bins)
    key_blocks: list = []
    n_clusters = len(clusters)
    for start in range(0, n_clusters, 2_000_000):
        by_len = {}
        for he in clusters[start:start + 2_000_000]:
            m = len(he)
            if m >= 2:                               # read_clusters de-dups loci -> all bins distinct
                by_len.setdefault(m, []).append(he)
        for m, group in by_len.items():
            arr = np.asarray(group, dtype=np.int64)  # (g, m), each row sorted ascending
            I, J = np.triu_indices(m, k=1)           # all i<j pairs -> row<col
            key_blocks.append((arr[:, I] * nb + arr[:, J]).ravel())

    if key_blocks:
        all_keys = np.concatenate(key_blocks)
        key_blocks.clear()
        del clusters                                 # free the concatemer list before the cooler build
        uk, counts = np.unique(all_keys, return_counts=True)   # sorted unique keys + their counts
        del all_keys
    else:
        uk = np.empty(0, dtype=np.int64)
        counts = np.empty(0, dtype=np.int64)

    rows = (uk // nb).astype(np.int64)
    cols = (uk % nb).astype(np.int64)
    del uk

    if build_mcool_file:
        bins = pd.DataFrame({
            "chrom": [f'chr{chrom + 1}' for chrom in chromosomes],
            "start": start_coords,
            "end": end_coords,
        })
        cooler_file = os.path.join(save_dir, "output.cool")
        mcool_file = os.path.join(save_dir, "output.mcool")

        def pixel_chunks(chunk: int = 50_000_000):
            # pixels are already sorted by (bin1_id, bin2_id) -> ordered=True
            for s in range(0, rows.shape[0], chunk):
                yield pd.DataFrame({
                    "bin1_id": rows[s:s + chunk],
                    "bin2_id": cols[s:s + chunk],
                    "count": counts[s:s + chunk].astype(np.int64),
                })

        print(f"Creating cool file ({rows.shape[0]:,} pixels) ...", flush=True)
        cooler.create_cooler(
            cooler_file, bins=bins, pixels=pixel_chunks(), ordered=True, assembly="hg38",
        )

        print("Zoomifying cool file...", flush=True)
        cooler.zoomify_cooler(
            base_uris=cooler_file,
            outfile=mcool_file,
            resolutions=[r for r in (1000, 5000, 10000, 25000, 50000, 100000, 250000, 1000000) if r >= resolution],
            chunksize=10_000_000,
            nproc=1,
            columns=["count"],
            dtypes={"count": "int64"},
            agg={"count": "sum"},
        )

        print("Balancing mcool file...", flush=True)
        with h5py.File(mcool_file, "r+") as f:
            for grp_name in list(f["resolutions"].keys()):
                uri = f"{mcool_file}::/resolutions/{grp_name}"
                print(f"Balancing {uri} ...", flush=True)
                cooler.balance_cooler(
                    cooler.Cooler(uri),
                    store=True  # persist weights to bins/weight (needed by toeplitz_normalize balance=True)
                )

def main(
    parent_dir: str,
    parent_save_dir: str,
    fname: str,
    resolution: int,
    create_mcool_files: bool = True
):
    
    file_path = os.path.join(parent_dir, fname)

    save_dir = os.path.join(parent_save_dir, os.path.basename(parent_dir))
    os.makedirs(save_dir, exist_ok=True)
    
    # Create list of hyperedges for inference
    clusters_for_inference = np.array(read_clusters(file_path, resolution), dtype=object)
    np.save(os.path.join(save_dir, 'edge_list.npy'), clusters_for_inference, allow_pickle=True)

    # Create contact map of virtual pairs -- only when actually requested. The clique-expansion
    # (a second read at base resolution + all-pairs over every concatemer) is unused by experiments
    # that supply real Hi-C, and at scale (e.g. 78M concatemers) it exhausts memory.
    if create_mcool_files:
        base_resolution = 1000
        clusters_for_virtual_hic = read_clusters(file_path, base_resolution)
        build_cooler_file(clusters_for_virtual_hic, base_resolution, create_mcool_files, save_dir)
    
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Process cluster files to generate contact maps and save clusters for inference.")
    argparser.add_argument('--parent-dir', type=str, required=True, help='Path to directory containing cluster files')
    argparser.add_argument('--parent-save-dir', type=str, required=True, help='Path to directory for saving processed files')
    argparser.add_argument('--resolution', type=int, default=100000, help='Resolution for inference (default: 100000)')    
    args = argparser.parse_args()

    for sub_dir in os.listdir(args.parent_dir):
        file_path = None
        sub_dir_path = os.path.join(args.parent_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file in os.listdir(sub_dir_path):
                if file.endswith('.clusters'):
                    print(file)
                    main(
                        parent_dir=sub_dir_path,
                        parent_save_dir=args.parent_save_dir,
                        fname=file,
                        resolution=args.resolution
                    )