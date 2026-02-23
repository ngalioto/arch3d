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

def toeplitz_normalize(
    matrix,
    chrom_slices
):
    
    n = chrom_slices[0][1] # largest chromosome size

    matrix_rows = np.repeat(np.arange(len(matrix.indptr)-1), np.diff(matrix.indptr))
    
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

    upper = scipy.sparse.triu(matrix, k=1)  # strictly upper triangular part (exclude diagonal)
    matrix = upper + upper.T + scipy.sparse.diags(matrix.diagonal())

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
    matrix = clr.matrix(sparse=True, balance=balance)[:].tocsr().astype(np.float64)

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
    parser = argparse.ArgumentParser(description='Compute Toeplitz normalization for a .mcool file.')
    parser.add_argument('filename')
    parser.add_argument('save_dir')
    parser.add_argument('--save_name')
    parser.add_argument('--weights_dir')
    parser.add_argument('--resolution', type=int, default=5000)
    parser.add_argument('--balance', type=bool, default=True)
    args = parser.parse_args()

    main(
        filename=args.filename, 
        save_dir=args.save_dir, 
        save_name=args.save_name,
        weights_dir=args.weights_dir,
        resolution=args.resolution,
        balance=args.balance
    )