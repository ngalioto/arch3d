import os
import scipy
import cooler
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import digitalcell.data.constants as constants

"""
Assumes the chromosomes are ordered from 1--22 in the .mcool file.
"""

def _toeplitz_normalize_legacy(
    matrix,
    chrom_slices
):
    # Reference implementation (per-chromosome-block loop) the vectorized toeplitz_normalize
    # below is validated byte-identical against. Kept for that validation; not called in prod.
    n = chrom_slices[0][1] # largest chromosome size

    # int32 row index (bin ids < 2^31): ~12GB lighter than int64 at GM's ~2.9B-nnz scale
    matrix_rows = np.repeat(np.arange(len(matrix.indptr)-1, dtype=np.int32), np.diff(matrix.indptr))
    
    pixels = np.zeros(n+1)
    counts = np.zeros(n+1)
    
    for ii in tqdm(range(22), position=0):
        chrom1 = chrom_slices[ii]
        row_size = chrom1[1] - chrom1[0]
    
        row_indices = np.logical_and(matrix_rows >= chrom1[0], matrix_rows < chrom1[1])
        
        for jj in tqdm(range(ii, 22), position=1, leave=False):
            chrom2 = chrom_slices[jj]
    
            col_indices = np.logical_and(matrix.indices >= chrom2[0], matrix.indices < chrom2[1])
    
            indices = np.logical_and(row_indices, col_indices)
    
            data = matrix.data[indices]
    
            # block is on the diagonal
            if ii == jj:
    
                diag_indices = np.abs(matrix_rows[indices] - matrix.indices[indices])
    
                np.add.at(pixels[:row_size], diag_indices, data)
                    
                np.add.at(counts[:row_size], diag_indices, np.ones(len(data)))
    
            # block is off the diagonal
            else:
                pixels[-1] += np.sum(data)
                counts[-1] += len(data)
    
    expected = np.zeros(n+1)
    np.divide(pixels, counts, out=expected, where=counts != 0)
    
    for ii in tqdm(range(22), position=0):
        chrom1 = chrom_slices[ii]
        row_size = chrom1[1] - chrom1[0]
    
        row_indices = np.logical_and(matrix_rows >= chrom1[0], matrix_rows < chrom1[1])
        
        for jj in tqdm(range(ii, 22), position=1, leave=False):
            chrom2 = chrom_slices[jj]
            col_size = chrom2[1] - chrom2[0]
            col_indices = np.logical_and(matrix.indices >= chrom2[0], matrix.indices < chrom2[1])
    
            indices = np.logical_and(row_indices, col_indices)
    
            # block is on the diagonal
            if ii == jj:
                # expected_row_indices are indices of diagonal block that corresponds to the data
                diag_indices = np.abs(matrix_rows[indices] - matrix.indices[indices])
    
                matrix.data[indices] /= expected[diag_indices]
    
    
            # block is off the diagonal
            else:
                matrix.data[indices] /= expected[-1]

    del matrix_rows  # free the per-nnz row index (~12GB at GM scale) before symmetrizing

    upper = scipy.sparse.triu(matrix, k=1)  # strictly upper triangular part (exclude diagonal)
    diag = scipy.sparse.diags(matrix.diagonal())
    matrix = upper + upper.T + diag

    return matrix, pixels, counts


def toeplitz_normalize(
    matrix,
    chrom_slices
):
    """Vectorized + sparse O/E, identical output to _toeplitz_normalize_legacy but in a SINGLE global
    pass: no 253x per-chromosome-block re-masking of all nonzeros, no np.add.at. Each nonzero is
    assigned to its chromosome once via searchsorted on the (contiguous) chrom-slice ends, the
    distance-expected is accumulated with np.bincount, and the division is vectorized. main()
    has already clipped the matrix to chr1-22, so every nonzero is in-autosome."""
    n = chrom_slices[0][1]                                          # largest chromosome size
    rows = np.repeat(np.arange(len(matrix.indptr) - 1, dtype=np.int32), np.diff(matrix.indptr))
    cols = matrix.indices
    ends = np.asarray([e for (_s, e) in chrom_slices], dtype=np.int64)   # increasing chrom-end bins
    row_chrom = np.searchsorted(ends, rows, side='right')          # chromosome index per nonzero
    col_chrom = np.searchsorted(ends, cols, side='right')
    cis = row_chrom == col_chrom
    dist = np.abs(rows.astype(np.int64) - cols.astype(np.int64))   # genomic-distance offset (cis)

    pixels = np.zeros(n + 1)
    counts = np.zeros(n + 1)
    dcis = dist[cis]
    pixels[:n] = np.bincount(dcis, weights=matrix.data[cis], minlength=n)[:n]   # cis by distance
    counts[:n] = np.bincount(dcis, minlength=n)[:n]
    trans = ~cis
    # The legacy loop accumulates trans from the UPPER block-triangle only (jj >= ii) and the
    # final triu keeps exactly those, so restrict the trans weights the same way for byte-identical
    # pixels/counts (the denominator is invariant to counting both symmetric copies, but the raw
    # saved weights are not). The division below still covers both copies -- harmless, triu discards
    # the lower ones.
    trans_up = trans & (row_chrom < col_chrom)
    pixels[-1] = matrix.data[trans_up].sum()                       # single trans bucket at index -1
    counts[-1] = int(trans_up.sum())

    expected = np.zeros(n + 1)
    np.divide(pixels, counts, out=expected, where=counts != 0)
    matrix.data[cis] /= expected[dcis]
    matrix.data[trans] /= expected[-1]

    upper = scipy.sparse.triu(matrix, k=1)
    diag = scipy.sparse.diags(matrix.diagonal())
    matrix = upper + upper.T + diag
    return matrix, pixels, counts


def main(
    filename: str,
    save_dir: str,
    save_name: str = None,
    weights_dir: str = None,
    resolution: int = 5000,
    balance: bool = True
):

    hic_id = os.path.splitext(os.path.basename(filename))[0]
    save_name = hic_id if save_name is None else save_name
    
    try:
        clr = cooler.Cooler(f"{filename}::/resolutions/{resolution}")
    except:
        try:
            resolution_index = constants.RESOLUTIONS.index(resolution)
            clr = cooler.Cooler(f"{filename}::{resolution_index}")
        except:
            try:
                clr = cooler.Cooler(f"{filename}") #ENCODE data
            except Exception as e:
                raise ValueError(f'ERROR {filename}: {e}')
                
    print('Extracting data from the cooler file...')
    matrix = clr.matrix(sparse=True, balance=balance)[:].tocsr().astype(np.float32)  # float32: halves the ~2.9B-nnz matrix

    if balance:
        np.nan_to_num(matrix.data, copy=False) # set NaNs to 0
        matrix.data[matrix.data > 1] = 0 # Set outliers to 0
        matrix.eliminate_zeros() # Remove zeros

    prefixes = ['chr', '']
    for prefix in prefixes:
        try:
            chrom_slices = [clr.extent(f'{prefix}{ii+1}') for ii in range(22)]
            break
        except ValueError:
            chrom_slices = None

    if chrom_slices is None:
        raise ValueError("Could not resolve chromosome naming convention in cooler file.")
        
    # clip matrix to chromosomes 1--22
    full_slice = slice(chrom_slices[0][0], chrom_slices[21][1])
    matrix = matrix[full_slice, full_slice]
    
    matrix, pixels, counts = toeplitz_normalize(matrix, chrom_slices)

    if weights_dir is not None:
        np.save(f'{weights_dir}/{save_name}_pixels.npy', pixels)
        np.save(f'{weights_dir}/{save_name}_counts.npy', counts)
        
    scipy.sparse.save_npz(f'{save_dir}/{save_name}.npz', matrix.astype(np.float32))

if __name__ == '__main__':
    # NOTE: argparse type=bool makes any non-empty string truthy (so --balance False
    # would still be True). Parse the string explicitly so False actually disables it.
    def _str2bool(v):
        return str(v).strip().lower() not in ('false', '0', 'no', 'f', 'n', '')

    parser = argparse.ArgumentParser(description='Compute Toeplitz normalization for a .mcool file.')
    parser.add_argument('filename')
    parser.add_argument('save_dir')
    parser.add_argument('--save_name')
    parser.add_argument('--weights_dir')
    parser.add_argument('--resolution', type=int, default=5000)
    parser.add_argument('--balance', type=_str2bool, default=True)
    args = parser.parse_args()

    main(
        filename=args.filename, 
        save_dir=args.save_dir, 
        save_name=args.save_name,
        weights_dir=args.weights_dir,
        resolution=args.resolution,
        balance=args.balance
    )