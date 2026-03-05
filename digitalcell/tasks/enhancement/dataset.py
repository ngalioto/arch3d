import os

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset

from digitalcell.data import constants
from digitalcell.data.dataset import (
    HiC_Data,
    HiC_Sequence,
    Locus_Position,
    aggregate_bins,
    get_resolution_map,
)

"""
A class for Hi-C data.
"""

def sample_pair_uniform_over_pairs(
    num_loci: int, 
    half_seq_len: int, 
    rng=None
):
    
    rng = np.random.default_rng(rng)
    if num_loci < 2*half_seq_len:
        raise ValueError("Need num_loci >= 2*half_seq_len.")
    M = num_loci - half_seq_len  # max start

    while True:
        a = rng.integers(0, M + 1)
        b = rng.integers(0, M + 1)
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        if lo + half_seq_len <= hi:              # non-overlap for [s, s+n)
            return lo, hi             # s1 < s2 always
    
class Enhance_HiC_Dataset(Dataset):

    def __init__(
        self,
        inputs_dir : Iterable[str],
        targets_dir : Iterable[str],
        seq_len: int = 1024,
        downsample_percentages: Iterable[int] = [1, 10, 100],
        resolutions: Iterable[int] = [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000],
        chroms: Iterable[int] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    ) -> None:
        
        """
        
        """
        
        super().__init__()

        self.resolutions = resolutions
        self.chroms = chroms
        self.seq_len = seq_len

        downsamples = [f"_{ds}pct" for ds in downsample_percentages]
        
        self.data = []
        self.refs = []

        for inputs, targets in zip(inputs_dir, targets_dir):
            input_data = []
            output_data = []
            for fname in os.listdir(inputs):
                if any(fname.endswith(f'{ds}.npz') for ds in downsamples) or (not fname.endswith('pct.npz') and fname.endswith('.npz')):
                    print(f'Loading input file: {fname}')
                    input_data.append(HiC_Data(
                        fname=os.path.join(inputs, fname),
                        chroms=chroms
                    ))
                else:
                    continue
            self.data.append(input_data) # nested list. Inner list has different coverages.
            for res in resolutions:
                # Find the file that ends with _{res}.npz in the targets directory
                target_files = [f for f in os.listdir(targets) if f.endswith(f'_{res}.npz')]
                if len(target_files) > 1:
                    raise ValueError(f'More than one target file found for resolution {res} in {targets}. Files found: {target_files}')
                if target_files:
                    print(f'Loading target file: {target_files[0]}')
                    try:
                        output_data.append(
                            scipy.sparse.load_npz(
                                os.path.join(targets, target_files[0])
                            )
                        )
                    except Exception as e:
                        print(f'Could not load file {os.path.join(targets, target_files[0])}.\n{e}')
                    np.clip(output_data[-1].data, a_min=0, a_max=15, out=output_data[-1].data)
            self.refs.append(output_data) # nested list. Inner list has different resolutions.

        self.num_inputs = [len(d) for d in self.data]
        self._len = len(resolutions) * sum(self.num_inputs)

    def __len__(
        self,
    ) -> int:
        
        """
        Returns the length of the dataset.

        Returns:
        --------
        int
            The length of the dataset.
        """

        return self._len

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Returns a HiC_Sequence object from the HiC_Data object at the given index.

        Parameters:
        -----------
        idx: int
            The index of the HiC_Data object to return the HiC_Sequence from.

        Returns:
        --------
        HiC_Sequence
            A HiC_Sequence object containing the loci positions and values required for the model.
        """

        items_per_exp = torch.tensor(self.num_inputs) * len(self.resolutions)
        exp_idx = torch.sum(idx >= torch.cumsum(items_per_exp, dim=0)).item()

        sub_idx = idx - torch.sum(items_per_exp[:exp_idx]).item()
        coverage_idx = sub_idx // len(self.resolutions)
        res_idx = sub_idx % len(self.resolutions)

        try:
            inputs = self.data[exp_idx][coverage_idx] # HiC_Data object
        except:
            raise IndexError(f'Index {idx} (yielding exp_idx={exp_idx}, coverage_idx={coverage_idx}) out of range for dataset with {self.__len__()} elements.')
        targets = self.refs[exp_idx][res_idx] # sparse matrix

        bins_per_locus = int(self.resolutions[res_idx] / constants.BASE_RESOLUTION)
        start_indices, stop_indices = get_resolution_map(bins_per_locus, self.chroms)
        half_seq_len = self.seq_len // 2
        lo, hi = sample_pair_uniform_over_pairs(
            num_loci=len(start_indices), 
            half_seq_len=half_seq_len,
            rng=None
        )
        bin_idx = torch.cat([torch.arange(lo, lo+half_seq_len), torch.arange(hi, hi+half_seq_len)])
        start, stop = start_indices[bin_idx], stop_indices[bin_idx]

        loci = aggregate_bins(
            inputs,
            start,
            stop,
            return_targets=False
        )

        input_loci = Locus_Position(
            start=constants.start_coords[None, start],
            end=constants.end_coords[None, stop-1], # subtract one because used for indexing now instead of slicing 
            chromosomes=constants.chromosomes[None, start],
            values=loci.unsqueeze(0)
        )
        hic_seq = HiC_Sequence(
            input_loci=input_loci
        )

        bin_idx = bin_idx.numpy()
        targets = torch.from_numpy(targets[bin_idx][:, bin_idx].toarray()).unsqueeze(0)
        
        return hic_seq, targets