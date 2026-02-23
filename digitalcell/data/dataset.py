import concurrent.futures
import math
import os
from functools import partial
from typing import Any, Iterable, Self, Tuple

import numpy as np
import scipy
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from digitalcell.data import constants

"""
A class for Hi-C data.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state",
    category=UserWarning,
    module="torch"
)

def is_contiguous_range(
    indices: torch.Tensor
) -> bool:
    
    """
    Checks if the given indices form a contiguous range.
    
    Parameters:
    -----------
    indices: Iterable[int]
        An iterable of indices (one-indexed)

    Returns:
    --------
    bool
        True if the indices form a contiguous range, False otherwise.
    """

    return torch.all(torch.diff(indices) == 1)

def get_chrom_indices(
    chroms: Iterable[int]
) -> torch.Tensor | slice:
    """
    Gets the row indices corresponding to the given chromosomes.

    Parameters:
    -----------
    chroms: Iterable[int]
        An iterable of chromosome numbers (one-indexed)

    Returns:
    --------
    np.ndarray
        An array of indices for all bins that fall into the chromosomes specified by `chroms`
    """

    # Number of bins across the genome at the given resolution
    if is_contiguous_range(chroms):
        chrom_start = constants.chrom_offset[chroms[0] - 1]
        chrom_end = constants.chrom_offset[chroms[-1]]
        chrom_idx = slice(chrom_start, chrom_end)
    else:
        chrom_idx = torch.cat([
            torch.arange(constants.chrom_offset[chrom - 1], constants.chrom_offset[chrom])
            for chrom in chroms
        ])
    return chrom_idx

def get_resolution_map(
    bins_per_locus: int = 20,
    chroms: Iterable[int] | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    NOTE: this can be combined with `tokenize_genome`
    """
    
    chroms = torch.arange(1, constants.NUM_CHROM + 1) if chroms is None else chroms

    start_indices = []
    stop_indices = []
    for chrom in chroms:
        start_indices.append(torch.arange(constants.chrom_offset[chrom - 1], constants.chrom_offset[chrom], bins_per_locus, dtype=torch.int32))
        stop_indices.append(torch.min(start_indices[-1] + bins_per_locus, constants.chrom_offset[chrom]))

    start_indices = torch.cat(start_indices)
    stop_indices = torch.cat(stop_indices)

    return start_indices, stop_indices


class Locus_Position:

    """
    Class containing all information needed to specify a "locus" on the genome
    """

    def __init__(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        chromosomes: torch.Tensor,
        values: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> None:
        
        """
        Initializes the Locus_Position object.

        B is batch size, N is sequence length, and num_bins is number of bins in the Hi-C data
        Parameters:
        -----------
        start: torch.Tensor
            The start positions of the loci. Shape: (B, N) dtype: torch.float32
        end: torch.Tensor
            The end positions of the loci (inclusive). Shape: (B, N) dtype: torch.float32
        chromosomes: torch.Tensor
            The chromosomes of the loci (zero-indexed). Shape: (B, N) dtype: torch.int32
        values: torch.Tensor
            The pixel values of the loci. For unmasked loci, this should contain the pixel values in the corresponding row of Hi-C. For masked loci, this should contain the pairwise pixel values for the pairwise contacts over all loci in the sequence. Shape: (B, N) or (B, N, num_bins), dtype: torch.float32
        mask: torch.Tensor
            The mask for loci with positive total reads. Shape: (B, N) or (B, N, num_bins), dtype: torch.bool
        """
        
        self.start = start
        self.end = end
        self.chromosomes = chromosomes
        self.values = values
        self.mask = mask
        self._len = start.shape[0]
        self._shape = start.shape

    def __len__(
        self
    ) -> int:

        """
        Returns the batch size of the Locus_Position object.
        """
        
        return self._len
    
    @property
    def shape(
        self
    ) -> Tuple[int, ...]:
        
        """
        Returns the shape of the Locus_Position object.
        """

        return self._shape
    
    @shape.setter
    def shape(
        self,
        value: Any
    ) -> None:
        
        """
        Prevents the shape attribute from being changed.

        Parameters:
        -----------
        value: Any
            The new shape value (not used)
        """

        raise AttributeError('The shape attribute cannot be changed.')
    
    def __getitem__(
        self,
        idx: int | slice
    ) -> Self:
        
        """
        Returns a Locus_Position object for the batch item(s) corresponding to the given index or slice.

        Parameters:
        -----------
        idx: int | slice
            The index or slice of the batch item(s)

        Returns:
        --------
        Locus_Position
            A new Locus_Position object containing only the batch item(s) corresponding to the given index or slice.
        """
        
        start = self.start[idx]
        end = self.end[idx]
        chromosomes = self.chromosomes[idx]
        values = self.values[idx] if self.values is not None else None
        mask = self.mask[idx] if self.mask is not None else None

        return Locus_Position(start, end, chromosomes, values, mask)

    def to(
        self,
        device: torch.device | str
    ) -> Self:

        """
        Moves the Locus_Position object to the given device.

        Parameters:
        -----------
        device: torch.device | str
            The device to move the tensors to

        Returns:
        --------
        Locus_Position
            A new Locus_Position object with all tensors moved to the given device.
        """

        return Locus_Position(
            start = self.start.to(device),
            end = self.end.to(device),
            chromosomes = self.chromosomes.to(device),
            values = self.values.to(device) if self.values is not None else None,
            mask = self.mask.to(device) if self.mask is not None else None
        )

class HiC_Sequence:

    """
    Class used as the input sequence for the Confound model.
    """

    def __init__(
        self,
        input_loci: Locus_Position,
        masked_loci: Locus_Position | None = None
    ) -> None:
        
        """
        Initializes the HiC_Sequence object.

        B is batch size, N is sequence length, and num_bins is number of bins in the Hi-C data
        Parameters:
        -----------
        input_loci: Locus_Position
            The unmasked loci in the input sequence. Shape: (B, N), dtype: torch.float32
        masked_loci: Locus_Position
            The masked loci in the input sequence. Shape: (B, N), dtype: torch.float32
        """
        
        self.input_loci = input_loci
        self.masked_loci = masked_loci

    def __len__(
        self
    ) -> int:

        """
        Returns the batch size of the HiC_Sequence object.

        Returns:
        --------
        int
            The batch size of the HiC_Sequence object.
        """
        
        return len(self.input_loci)
    
    @property
    def shape(
        self
    ) -> Tuple[int, ...]:
        
        """
        Returns the shape of the HiC_Sequence object.

        Returns:
        --------
        Tuple[int, ...]
            The shape of the HiC_Sequence object. (B, N)
        """
        
        return self.input_loci.shape
    
    @shape.setter
    def shape(
        self,
        value: Any
    ) -> None:
        
        raise AttributeError('The shape attribute cannot be changed.')
    
    def __getitem__(
        self,
        idx: int | slice
    ) -> Self:
        
        """
        Returns a HiC_Sequence object for the batch item(s) corresponding to the given index or slice.
        
        Parameters:
        -----------
        idx: int | slice
            The index or slice of the batch item(s)

        Returns:
        --------
        HiC_Sequence
            A new HiC_Sequence object containing only the batch item(s) corresponding to the given index or slice.
        """
        
        input_loci = self.input_loci[idx]
        masked_loci = self.masked_loci[idx] if self.masked_loci is not None else None

        return HiC_Sequence(
            input_loci=input_loci,
            masked_loci=masked_loci
        )

    def to(
        self,
        device: torch.device | str
    ) -> Self:
        
        """
        Moves the HiC_Sequence object to the given device.

        Parameters:
        -----------
        device: torch.device | str
            The device to move the tensors to

        Returns:
        --------
        HiC_Sequence
            A new HiC_Sequence object with all tensors moved to the given device.
        """

        return HiC_Sequence(
            input_loci = self.input_loci.to(device),
            masked_loci = self.masked_loci.to(device) if self.masked_loci is not None else None
        )

class HiC_Data:

    """
    Class containing the Hi-C contact matrix and associated metadata.
    """
    
    def __init__(
        self,
        fname: str,
        chroms: Iterable = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    ) -> None:

        """
        Initializes the HiC_Data object.

        Parameters:
        -----------
        fname: str
            The path to the .npz file containing the Hi-C contact matrix.
        chroms: Iterable
            The chromosomes to include in the Hi-C contact matrix (one-indexed).
        """

        self.chroms = torch.tensor(chroms) if not isinstance(chroms, torch.Tensor) else chroms

        chrom_idx = get_chrom_indices(self.chroms)

        self._fname = f'{os.path.splitext(fname)[0]}.npz'
        self.file_size = os.path.getsize(self.fname)
        
        self._loci = scipy.sparse.load_npz(self.fname)[chrom_idx, :constants.NUM_BINS]

        self.num_bins = self._loci.shape[0]  # number of rows after subsetting chromosomes

        np.clip(self._loci.data, a_min=0, a_max=15, out=self._loci.data)
        

    @property
    def fname(
        self
    ) -> str:
        
        """
        Returns the filename of the Hi-C data.
        
        Returns:
        --------
        str
            The filename of the Hi-C data.
        """
        
        return self._fname

    @fname.setter
    def fname(
        self,
        value: Any
    ) -> None:
        
        """
        Prevents the filename from being changed.
        
        Parameters:
        -----------
        value: Any
            The new filename value (not used)
        """
        
        raise AttributeError('The filename cannot be changed. A new HiC_Data object must be created.')

    @property
    def loci(
        self
    ) -> csr_matrix:
        
        """
        Returns the Hi-C contact matrix as a scipy sparse matrix.

        Returns:
        --------
        csr_matrix
            The Hi-C contact matrix as a scipy sparse matrix.
        """
    
        return self._loci
    
    @loci.setter
    def loci(
        self,
        value: Any
    ) -> None:
        
        """
        Prevents the loci data from being changed.

        Parameters:
        -----------
        value: Any
            The new loci value (not used)
        """

        self._loci = value
        # raise AttributeError('The loci data cannot be changed.')

    def __repr__(
        self
    ) -> str:
        
        """
        Returns a string representation of the HiC_Data object.
        
        Returns:
        --------
        str
            A string representation specifying the filename and file size of the HiC_Data object.
        """
            
        return f'<HiC_Data object. fname: {self.fname}, size: {self.file_size / (1024 * 1024 * 1024)} GB>'

    def sample_loci(
        self,
        num_samples: int,
        loci_sampler_vals: torch.Tensor | None = None,
        loci_sampler_dist: torch.distributions.Distribution | None = None,
        shuffle: bool = True,
        fixed_grid: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Randomly samples loci from the genome. The loci are returned ordered across the genome unless shuffle is set to True.

        Parameters:
        -----------
        num_samples: int
            The number of locus samples to draw
        loci_sampler_vals: torch.Tensor
            A tensor of possible locus lengths to sample from
        loci_sampler_dist: torch.distributions.Distribution
            A distribution over the possible locus lengths
        shuffle: bool
            Whether to shuffle the loci before returning them
        fixed_grid: bool
            If true, uses a single resolution grid for sampling loci. If false, loci of varying lengths can be selected.

        Returns:
        --------
        start_indices: torch.Tensor
            The starting indices of the sampled loci
        stop_indices: torch.Tensor
            The ending indices of the sampled loci (exclusive)
        """

        if fixed_grid:  # Calculate the step between the starting points of each interval

            if loci_sampler_vals is None:
                interval_length = 20 # 100 kb by default
            elif loci_sampler_vals.ndim == 0:
                interval_length = loci_sampler_vals.item()
            elif len(loci_sampler_vals) == 1:
                interval_length = loci_sampler_vals[0]
            else:
                raise ValueError('Fixed grid sampling only works for single locus length.')

            start_indices, stop_indices = get_resolution_map(interval_length, self.chroms)

            if len(start_indices) < num_samples:
                raise ValueError(f'Not enough loci for fixed grid sampling with the given locus length. The total number of loci is {len(start_indices)}, but {num_samples} samples were requested.')

            bin_idx = torch.randperm(len(start_indices))[:num_samples]
            if not shuffle:
                bin_idx, _ = torch.sort(bin_idx)

            return start_indices[bin_idx], stop_indices[bin_idx]

        else:

            if loci_sampler_vals is not None and loci_sampler_dist is not None:
                
                loci_lengths = loci_sampler_vals[loci_sampler_dist.sample((num_samples,))].to(torch.int32)
                total_gap_length = self.num_bins - loci_lengths.sum(dim=-1)

                loci_cumsum = torch.cumsum(torch.cat((torch.zeros(1, dtype=torch.int32), loci_lengths[:-1])), dim=0).int()

                if total_gap_length < 0:
                    raise ValueError('The sampled loci exceed the length of the human genome.')
                elif total_gap_length == 0:
                    start_indices = loci_cumsum[:-1]
                    stop_indices = loci_cumsum[1:]
                else:
                    # sample lengths of gaps between loci
                    gap_lengths = torch.rand(len(loci_lengths) + 1)
                    # make gaps add up to the total gap length
                    gap_lengths *= total_gap_length / torch.sum(gap_lengths)
                    gap_lengths = torch.floor(gap_lengths).int()
                    for ii in range(total_gap_length - torch.sum(gap_lengths)):
                        gap_lengths[ii] += 1

                    start_indices = torch.cumsum(gap_lengths[:-1], dim=0, dtype=torch.int32) + loci_cumsum
                    stop_indices = start_indices + loci_lengths

            elif loci_sampler_vals is None and loci_sampler_dist is None:
                # randomly sample singular rows
                start_indices = torch.randperm(self.num_bins)[:num_samples]
                if not shuffle:
                    start_indices, _ = torch.sort(start_indices)
                stop_indices = start_indices + 1
                return start_indices, stop_indices
            
            else:
                raise ValueError('Both loci_sampler_vals and loci_sampler_dist must be provided, or neither, when `fixed_grid` is False.')

        if shuffle:
            # shuffle indices
            idx_shuffle = torch.randperm(len(start_indices))
            return start_indices[idx_shuffle], stop_indices[idx_shuffle]
        else:
            return start_indices, stop_indices
    
class HiC_Dataset(Dataset):

    def __init__(
        self,
        data_list: Iterable[str] | None = None,
        max_len: int = 1024,
        loci_sampler_vals: torch.Tensor | None = None,
        loci_sampler_probs: torch.Tensor | None = None,
        batch: Iterable[HiC_Data] | None = None,
        num_masked: int | None = None,
        num_unchanged: int | None = None,
        fixed_grid: bool = False,
        max_workers: int | None = None,
        chroms: Iterable[int] | None = None
    ) -> None:
        
        """
        Initializes the HiC_Dataset object.

        Parameters:
        -----------
        data_list: Iterable[str]
            An iterable of paths to .npz files containing Hi-C contact matrices.
        max_len: int
            The maximum length of the encoder input sequence.
        loci_sampler_vals: torch.Tensor
            A tensor of possible locus lengths to sample from.
        loci_sampler_probs: torch.Tensor
            A tensor of probabilities for the locus lengths to sample from.
        batch: Iterable[HiC_Data]
            An iterable of HiC_Data objects to include in the dataset.
        num_masked: int
            The number of loci to mask in the input sequence.
        num_unchanged: int
            The number of loci to leave unchanged in the input sequence.
        fixed_grid: bool
            Whether to use the same loci every time.
        max_workers: int
            The maximum number of workers to use for loading the data.
        chroms: Iterable[int]
            The chromosomes to include in the Hi-C contact matrices (one-indexed).
        
        Note: Either `data_list` or `batch` must be provided. If both are provided, the datasets will be combined.
        """
        
        super().__init__()

        self.max_len = max_len
        self.loci_sampler_vals = loci_sampler_vals if isinstance(loci_sampler_vals, torch.Tensor) else torch.tensor(loci_sampler_vals)
        if loci_sampler_vals is not None:
            if loci_sampler_probs is None:
                self.loci_sampler_probs = torch.ones(len(loci_sampler_vals)) # uniform distribution
            else:
                self.loci_sampler_probs = loci_sampler_probs if isinstance(loci_sampler_probs, torch.Tensor) else torch.tensor(loci_sampler_probs)
            self.loci_sampler_dist = torch.distributions.Categorical(self.loci_sampler_probs)
        else:
            self.loci_sampler_dist = None
            self.loci_sampler_probs = None
        self.num_masked = num_masked
        self.num_unchanged = num_unchanged
        self.fixed_grid = fixed_grid

        if batch is not None:
            self.data = [hic_data for hic_data in batch]
        else:
            self.data = []
            
        if data_list is not None:

            # Use process pool for parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a partial function with the chroms parameter
                process_func = partial(process_file, chroms=chroms)

                # Process files in parallel and collect results
                results = list(executor.map(process_func, data_list))

                # Filter out None results
                print('Filtering results')
                valid_results = [r for r in results if r is not None]
                self.data.extend(valid_results)

            print(f'Found {len(self.data)} files')
            

    def get_experiment(
        self,
        idx: int | slice,
    ) -> HiC_Data | list[HiC_Data]:
        
        """
        Gives the HiC_Data object at the given index.

        Parameters:
        -----------
        idx: int | slice
            The index or slice of the HiC_Data object(s) to return.

        Returns:
        --------
        HiC_Data | list[HiC_Data]
            The HiC_Data object(s) at the given index or slice.
        """
        
        return self.data[idx]

    def __getitem__(
        self,
        idx: int,
    ) -> HiC_Sequence:
        
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        hic_data = self.get_experiment(idx)
        return process_item(
            data=hic_data, 
            max_len=self.max_len,
            loci_sampler_vals=self.loci_sampler_vals,
            loci_sampler_dist=self.loci_sampler_dist,
            num_masked=self.num_masked,
            num_unchanged=self.num_unchanged, 
            device=device
        )
        
    def random_split(
        self,
        split: Iterable[float | int],
        generator: torch.Generator
    ) -> list[Self]:
        
        """
        Randomly splits the dataset into non-overlapping new datasets for training, validation, and testing.
        """
        
        num_data = len(self)
        total = sum(split)
        if total != 1 and total != num_data:
            raise ValueError('Split must sum to 1 or the length of the dataset')

        random_idx = torch.randperm(num_data, generator=generator)

        if total == 1:
            split = np.array([int(num_data * frac) for frac in split])
            remainder = num_data - split.sum()
            split[:remainder] += 1

        split_idx = np.concatenate((np.zeros((1,), dtype=int), np.cumsum(split)))
        assert sum(split) == num_data, "Error in forming the train/val/test split numbers"

        data_split = [
            HiC_Dataset(
                batch=[self.get_experiment(int(ii)) for ii in random_idx[split_idx[jj]:split_idx[jj+1]]],
                max_len = self.max_len,
                loci_sampler_vals = self.loci_sampler_vals,
                loci_sampler_probs = self.loci_sampler_probs,
                num_masked = self.num_masked,
                num_unchanged = self.num_unchanged
            )
        for jj in range(len(split))]

        return data_split

    
    def __len__(
        self
    ) -> int:
        
        return len(self.data)
    
    def __repr__(
        self
    ) -> str:
        
        return f'<HiC_Dataset object containing {len(self)} files>'
    
    

    

def tokenize_chrom(
    data: HiC_Data,
    num_row_bins: int,
    chrom: int,
    max_len: int | None = None,
    shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Used in many of the downstream testing and tasks. Need to make more generalizable, specifically `chrom_end`
    
    chrom is one-based
    """
    
    if chrom not in data.chroms:
        raise ValueError(f'Chromosome {chrom} was passed, but the Hi-C data only has chromosomes {data.chroms}.')

    # starting and ending bin on the full genome
    num_bins = constants.chrom_offset[chrom] - constants.chrom_offset[chrom - 1]
    
    num_tokens = math.ceil(num_bins / num_row_bins)
    if max_len is not None:
        num_tokens = min(max_len, num_tokens)
    
    # beginning bin of each token (from zero)
    start = torch.arange(0, num_row_bins * num_tokens, num_row_bins, dtype=torch.int32)
    
    # ending bin of each token (from zero)
    stop = torch.zeros_like(start)
    stop[:-1] = torch.arange(num_row_bins, num_row_bins * num_tokens, num_row_bins)
    stop[-1] = min(num_bins, num_row_bins * num_tokens) # min because last token could be smaller than num_row_bins
    
    rel_pos = torch.arange(len(data.chroms))[data.chroms == chrom]
    if rel_pos > 0:
        chrom_lengths = constants.chrom_offset[1:] - constants.chrom_offset[:-1]
        chrom_offset = torch.sum(chrom_lengths[data.chroms[:rel_pos] - 1], dtype=torch.int32)

        start += chrom_offset
        stop += chrom_offset

    if shuffle:
        # shuffle indices
        idx_shuffle = torch.randperm(len(start))
        start = start[idx_shuffle]
        stop = stop[idx_shuffle]

    return start, stop


def tokenize_genome(
    data: HiC_Data,
    num_row_bins: int,
    chroms: Iterable = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
    max_len: int | None = None,
    shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Partitions the whole genome into loci of size `num_row_bins` * 5kb.
    """
    
    max_len = data.num_bins if max_len is None else max_len
    
    total_len = 0
    total_start = []; total_stop = []
    for chrom in chroms:
        start, stop = tokenize_chrom(data, num_row_bins, chrom)
        total_len += len(start)
        total_start.append(start); total_stop.append(stop)
        if not shuffle and total_len >= max_len:
            break
    start = torch.cat(total_start); stop = torch.cat(total_stop)
    
    if shuffle:
        # shuffle indices
        idx_shuffle = torch.randperm(len(start))
        start = start[idx_shuffle]
        stop = stop[idx_shuffle]

    return start[:max_len], stop[:max_len]


def process_item(
    data: HiC_Data,
    max_len: int,
    loci_sampler_vals: torch.Tensor,
    loci_sampler_dist: torch.distributions.Distribution,
    num_masked: int = 0,
    num_unchanged: int | None = None,
    fixed_grid: bool = False,
    device: torch.device | str = 'cpu'
) -> HiC_Sequence:
    
    """

    Assumes that chromosomes 1-22 the first 22 chromosomes in the matrix

    This function extracts everything needed to embed a Hi-C experiment and compute the loss function from a HiC_Data object.

    Parameters:
    -----------
    data: HiC_Data
        The Hi-C data object
    max_len: int
        The maximum length of the encoder input sequence
    loci_sampler_vals: torch.Tensor
        The values to sample from the loci_sampler_dist
    loci_sampler_dist: torch.distributions.Distribution
        The distribution to sample from
    num_masked: int
        The number of loci to mask
    num_unchanged: int | None
        The number of unmasked loci to enter into the loss. If None, all unmasked loci are used.
    fixed_grid: bool
        Whether to use the same loci every time
    device: torch.device | str
        The device to move the tensors to

    Returns
    -------
    HiC_Sequence
        The HiC_Sequence object containing the loci positions and values required for the model
    """

    # sample start and stop indices for loci
    start, stop = data.sample_loci(max_len, loci_sampler_vals, loci_sampler_dist, shuffle=False, fixed_grid=fixed_grid)
    # loci spanning two chromosomes are placed on a single chromosome
    start, stop = truncate_loci(start, stop)

    hic_seq = tokenize_data(data, start, stop, num_masked, num_unchanged, device=device)

    return hic_seq

def get_masking_masks(
    n_seq: int,
    num_masked: int,
    num_predicted: int
) -> tuple[torch.Tensor, torch.Tensor]:

    prediction_idx = torch.randperm(n_seq)[:num_predicted]
    masking_idx = prediction_idx[:num_masked]

    masking_mask = torch.zeros(n_seq, dtype=torch.bool)
    prediction_mask = torch.zeros(n_seq, dtype=torch.bool)
    masking_mask[masking_idx] = True
    prediction_mask[prediction_idx] = True

    return masking_mask, prediction_mask

def aggregate_bins(
    data: HiC_Data,
    start: torch.Tensor,
    stop: torch.Tensor,
    return_input_loci: bool = True,
    return_targets: bool = True,
    device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    
    # if device.startswith("cuda"):
    #     return aggregate_bins_with_gpu(data, start, stop, return_input_loci, return_targets, device)
    
    n_seq = len(start)
    
    # row_map maps the rows of the original resolution matrix to the rows of the multi-resolution matrix
    row_map = torch.repeat_interleave(torch.arange(n_seq, dtype=torch.int32), stop - start)

    # num_rows are the number of original resolution rows we are extracting
    num_rows = len(row_map)

    # indices are the indices of the rows to extract from the orignal resolution matrix (must be numpy for indexing sparse matrix)
    row_indices = np.concatenate([np.arange(start_ii, stop_ii, dtype=np.int32) for start_ii, stop_ii in zip(start, stop)])
    col_indices = row_indices + constants.chrom_offset[data.chroms[0] - 1].numpy()

    scalars = torch.ones(num_rows, dtype=torch.int32)

    # sums over rows within each genomic interval
    S = csr_matrix((scalars, (row_map, np.arange(num_rows))), shape=(n_seq, num_rows))
    
    # removes a submatrix of the full Hi-C map containing just the sampled rows
    loci = data.loci[row_indices]

    # Get the number of nonzeros in every column of every locus
    # row_nnz = loci.astype(bool).astype(int)
    row_nnz = loci.sign() # 1. if >0, and 0. if 0
    row_nnz = S @ row_nnz
    
    loci = S @ loci

    # These are the inputs that the linear projection acts on
    if return_input_loci:
        input_loci = loci.copy()
        input_loci.data /= row_nnz.data
        input_loci = torch.tensor(input_loci.todense()).float()
    
    # These are the targets for the loss
    if return_targets:
        targets = loci[:, col_indices] @ S.T
        row_nnz = row_nnz[:, col_indices] @ S.T
        targets.data /= row_nnz.data
        targets = torch.tensor(targets.todense()).float()
    
    if return_input_loci and return_targets:
        return input_loci, targets
    elif return_input_loci and not return_targets:
        return input_loci
    elif not return_input_loci and return_targets:
        return targets
    

def tokenize_data(
    data: HiC_Data,
    start: torch.Tensor,
    stop: torch.Tensor,
    num_masked: int | None = None,
    num_unchanged: int | None = None,
    device: torch.device | str = 'cpu'
) -> HiC_Sequence:
    
    """
    Extracts everything needed to embed a Hi-C experiment and compute the loss function from a HiC_Data object.

    Parameters:
    -----------
    data: HiC_Data
        The Hi-C data object
    start: torch.Tensor
        The starting indices of the loci. Shape: (N,)
    stop: torch.Tensor
        The ending indices of the loci (exclusive). Shape: (N,)
    num_masked: int | None
        The number of loci to mask. If None, `masked_loci` object will be set to None.
    num_unchanged: int | None
        The number of unmasked loci to enter into the loss. If None, the maximum number of unmasked loci will be used.
    device: torch.device | str
        The device to place the tensors on.
        
    Returns
    -------
    HiC_Sequence
        The HiC_Sequence object containing the loci positions and values required for the model. If num_masked=0, the masked_loci object will be None.
    """
    
    n_seq = len(start)
    
    create_masked_loci = num_masked is not None
    num_masked = 0 if num_masked is None else num_masked
    
    # argument checking
    if num_masked > n_seq:
        raise ValueError(f'The number of masked loci must be strictly less than the total number of loci, but got values {num_masked} and {n_seq}, respectively.')
        
    if num_unchanged is None:
        num_unchanged = n_seq - num_masked
    
    loci, targets = aggregate_bins(data, start, stop, device=device)
    
    num_predicted = min(n_seq, num_masked + num_unchanged)
    masking_mask, prediction_mask = get_masking_masks(n_seq, num_masked, num_predicted)
    unmasked_mask = ~masking_mask

    # The values here are the inputs to the model
    input_loci = Locus_Position(
        start=constants.start_coords[None, start[unmasked_mask]],
        end=constants.end_coords[None, stop[unmasked_mask]-1], # subtract one because used for indexing now instead of slicing 
        chromosomes=constants.chromosomes[None, start[unmasked_mask]], 
        mask=unmasked_mask[None, :],
        values=loci[None, :, :]
    )

    # The values here will be used as the targets in the loss
    masked_loci = Locus_Position(
        start=constants.start_coords[None, start[masking_mask]], 
        end=constants.end_coords[None, stop[masking_mask]-1], # subtract one because used for indexing now instead of slicing 
        chromosomes=constants.chromosomes[None, start[masking_mask]], 
        mask=prediction_mask[None, :],
        values=targets[prediction_mask][None, :, prediction_mask]
    )  if create_masked_loci else None
        

    hic_seq = HiC_Sequence(
        input_loci = input_loci,
        masked_loci = masked_loci
    )

    return hic_seq.to(device)

def truncate_loci(
    start: torch.Tensor,
    end: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Truncates the loci so that each one falls only on a single chromosome.
    If a loci crosses two chromosomes, only the largest single-chromosome segment is kept

    Parameters:
    -----------
    start: torch.Tensor
        Indices corresponding to the start indices of the loci. Shape: (N,)
    end: torch.Tensor
        Indices corresponding to the end indices of the loci (exclusive). Shape: (N,)

    Returns:
    --------
    start:
        The corrected start indices. Shape: (N,)
    end:
        The corrected end indices (exclusive). Shape: (N,)
    """  

    # indices of starting chromosomes (zero-based)
    start_chrom = constants.chromosomes[start]
    # indices of ending chromosomes (zero-based)
    end_chrom = constants.chromosomes[end - 1] # -1 because end is exclusive
    multi_chrom_loci = start_chrom != end_chrom

    if multi_chrom_loci.sum() > 0:
        # point where the second chromosome starts
        chrom_boundary = constants.chrom_offset[end_chrom[multi_chrom_loci]]

        # subtract starting locus position from ending chrom position
        chrom1_len = chrom_boundary - start[multi_chrom_loci]
        # subtract starting chrom position from end locus position
        chrom2_len = end[multi_chrom_loci] - chrom_boundary

        chrom1_is_longer = chrom1_len > chrom2_len

        # If chrom1 is longer, then locus ends on chrom boundary
        end[multi_chrom_loci][chrom1_is_longer] = chrom_boundary[chrom1_is_longer]
        # If chrom2 is longer, then locus starts on chrom boundary
        start[multi_chrom_loci][~chrom1_is_longer] = chrom_boundary[~chrom1_is_longer]

    return start, end

def process_file(
    array_path: str, 
    chroms: Iterable[int]
) -> HiC_Data | None:

    experiment_id, ext = os.path.splitext(array_path)[:2]

    # Process only NPZ files
    if ext == '.npz':
        try:
            hic_data = HiC_Data(
                fname=array_path,
                chroms=chroms
            )

            print(f'Hi-C {experiment_id} added')
            return hic_data
        except Exception as e:
            print(f"Error processing {array_path}: {str(e)}")
            return None

    return None

###########################
# Functions for collation #
###########################

def stack_locus_positions(
    loci: Iterable[Locus_Position]
) -> Locus_Position:
    
    """
    Stacks a list of Locus_Position objects into a single Locus_Position object.

    Parameters:
    -----------
    loci: Iterable[Locus_Position]
        Iterable of Locus_Position objects

    Returns
    -------
    locus: Locus_Position
        Single Locus_Position object where the attributes are stacked along the first dimension
    """
    
    start = torch.cat([locus.start for locus in loci])
    end = torch.cat([locus.end for locus in loci])
    chromosomes = torch.cat([locus.chromosomes for locus in loci])
    values = torch.cat([locus.values for locus in loci]) if loci[0].values is not None else None
    mask = torch.cat([locus.mask for locus in loci]) if loci[0].mask is not None else None

    return Locus_Position(start, end, chromosomes, values, mask)

def collate_hic_sequences(
    batch: Iterable[HiC_Sequence]
) -> HiC_Sequence:
    
    """
    Collates a list of HiC_Sequence objects into a single HiC_Sequence object.

    Assumes that all sequences were generated with the same hyperparameters.

    Parameters:
    -----------
    batch: Iterable[HiC_Sequence]
        Non-empty iterable of HiC_Sequence objects

    Returns
    -------
    sequence: HiC_Sequence
        Single HiC_Sequence object where the attributes are stacked along the first dimension
    """
    
    input_loci = stack_locus_positions([sequence.input_loci for sequence in batch])
    masked_loci = stack_locus_positions([sequence.masked_loci for sequence in batch]) if batch[0].masked_loci is not None else None

    return HiC_Sequence(
        input_loci = input_loci,
        masked_loci = masked_loci
    )