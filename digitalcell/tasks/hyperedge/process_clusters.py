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

    chrom_offset = constants.get_chrom_sizes()
    chrom_bins = torch.ceil(chrom_offset / resolution).int()
    chrom_offset = torch.cat((torch.zeros(1, dtype=torch.int32), chrom_bins.cumsum(dim=0, dtype=torch.int32)))

    clusters = []
    LOCUS_RE = re.compile(r"^chr([0-9]+):(\d+)$")
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()            # splits on tabs or multiple spaces
            cluster_id, tokens = parts[0], parts[1:]
            # start_loci = []; stop_loci = []
            loci = []
            for t in tokens:
                m = LOCUS_RE.match(t)
                if m:
                    chrom, pos = int(m.group(1)), int(m.group(2))
                    
                    # start, stop = basepair_to_bin(chrom, pos)
                    # start_loci.append(start); stop_loci.append(stop)
                    bin_idx = basepair_to_bin(chrom, pos, chrom_offset, resolution)
                    if bin_idx is not None:
                        loci.append(bin_idx)
                    
            # clusters.append((torch.tensor(start_loci), torch.tensor(stop_loci)))
            # CONSIDER: clusters.append(torch.cat(loci))
            clusters.append(sorted(set(loci))) # list of hyperedges. Each hyperedge is list of bins
            # each hyperedge must be sorted for `generate_kmers.py` to work properly
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
    
    rows = []; cols = []
    for hyperedge in clusters:
        edge_order = len(hyperedge)
        for ii in range(edge_order):
            for jj in range(ii+1, edge_order):
                if hyperedge[ii] == hyperedge[jj]:
                    continue
                else:
                    rows.append(hyperedge[ii])
                    cols.append(hyperedge[jj])
                
    rows = np.array(rows).reshape(-1)
    cols = np.array(cols).reshape(-1)

    data = np.ones_like(rows, dtype=np.float64)
    hic = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_bins, num_bins))
    hic.sum_duplicates()
    if build_mcool_file:
        # Cooler likes sorted pixels
        bin1 = hic.row.astype(np.int64)
        bin2 = hic.col.astype(np.int64)
        order = np.lexsort((bin1, bin2))
        pixels = pd.DataFrame({
            "bin1_id": bin1[order],
            "bin2_id": bin2[order],
            "count": hic.data.astype(np.int64)[order]
        })
        bins = pd.DataFrame({
            "chrom": [f'chr{chrom + 1}' for chrom in chromosomes],
            "start": start_coords,
            "end": end_coords
        })
        pixels.to_csv(os.path.join(save_dir, "matrix.txt"), sep="\t", header=False, index=False)
        bins.to_csv(os.path.join(save_dir, "bins.bed"), sep="\t", header=False, index=False)
        
        print("Creating cool file...")
        cooler_file = os.path.join(save_dir, "output.cool")
        mcool_file = os.path.join(save_dir, "output.mcool")
        cooler.create_cooler(
            cooler_file,
            bins=bins,
            pixels=pixels,
            assembly="hg38"
        )

        print("Zoomifying cool file...")
        cooler.zoomify_cooler(
            base_uris=cooler_file,      # or a list of base coolers
            outfile=mcool_file,       # where to write the multires file
            resolutions=[1000, 5000, 10000, 25000, 50000, 100000, 250000, 1000000],
            chunksize=10_000_000,         # number of pixels per worker chunk (tune for memory)
            nproc=1,                      # >1 will multiprocessing coarsen in parallel
            columns=["count"],            # which pixel columns to propagate
            dtypes={"count": "int64"},    # dtype for that column in each zoom level
            agg={"count": "sum"},         # how to aggregate values when coarsening
        )

        print("Balancing mcool file...")
        with h5py.File(mcool_file, "r+") as f:
            # list the internal groups that are actual coolers
            for grp_name in list(f["resolutions"].keys()):
                uri = f"{mcool_file}::/resolutions/{grp_name}"
                clr = cooler.Cooler(uri)
                print(f"Balancing {uri} ...")

                cooler.balance_cooler(
                    clr
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

    # Create contact map of virtual pairs
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