import threading
import torch
import pandas as pd
import os
import sys

# Static constants
NUM_CHROM = 22
RESOLUTIONS = [20480000, 10240000, 5120000, 2560000, 1280000, 640000, 320000, 160000, 80000, 40000, 20000, 10000, 5000]
NUM_BINS = 575010
BASE_RESOLUTION = 5000
REF_DIR = os.path.join(os.path.dirname(__file__), 'ref')

# Lazy-loaded attributes
_LAZY_ATTRS = {'num_bins', 'chrom_offset', 'chromosomes', 'start_coords', 'end_coords'}

# Thread-safe initialization
_initialized = False
_init_lock = threading.Lock()

def _download_chrom_sizes(dir_path: str) -> None:
    import requests

    os.makedirs(dir_path, exist_ok=True)
    url = "https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes"
    file_name = os.path.join(dir_path, "hg38.chrom.sizes")

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete → {file_name}")
    else:
        print(f"Error: {response.status_code}")


def get_chrom_sizes() -> pd.DataFrame:
    """
    Retrieve chromosome sizes from the reference directory.

    Returns:
        pd.DataFrame: DataFrame containing chromosome sizes.
    """
    hg38_path = os.path.join(REF_DIR, "hg38.chrom.sizes")
    if not os.path.exists(hg38_path):
        print('Could not find genome reference file "hg38.chrom.sizes". Downloading from UCSC...')
        _download_chrom_sizes(REF_DIR)

    chrom_sizes = pd.read_csv(hg38_path, sep="\t", header=None, names=["chrom", "size"])
    chrom_sizes = torch.tensor([chrom_sizes.loc[chrom_sizes['chrom'] == f'chr{ii}', 'size'].iloc[0]
        for ii in range(1, NUM_CHROM + 1)
    ], dtype=torch.int32)

    return chrom_sizes

def get_chrom_offset(
    resolution: int
):

    """
    Compute chromosome offsets based on the given resolution.

    Parameters:
    ----------
    resolution : int
        Resolution for binning.

    Returns:
    -------
    torch.Tensor
        Chromosome offsets tensor.
    """
    chrom_sizes = get_chrom_sizes()
    chrom_bins = torch.ceil(chrom_sizes / resolution).int()
    chrom_offset = torch.cat((torch.zeros(1, dtype=torch.int32), chrom_bins.cumsum(dim=0, dtype=torch.int32)))
    
    return chrom_offset

def compute_bins_coordinates(
    resolution: int
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the bins coordinates based on the chromosome sizes.
    
    chromosomes is zero-based
    """
        
    chrom_sizes = get_chrom_sizes()

    lengths = torch.ceil(chrom_sizes / resolution).int()
    chrom_offset = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(dim=0, dtype=torch.int32)))
    chromosomes = torch.repeat_interleave(torch.arange(NUM_CHROM, dtype=torch.int32), lengths)

    num_bins = int(chrom_offset[-1])

    start_coords = [torch.arange(0, chrom_sizes[chrom], resolution, dtype=torch.float32) for chrom in range(NUM_CHROM)]
    end_coords = [torch.min(start_coords[chrom] + resolution, chrom_sizes[chrom]) for chrom in range(NUM_CHROM)]

    scaling_factor = 1e-6
    start_coords = scaling_factor * torch.cat(start_coords, dim=0)
    end_coords = scaling_factor * torch.cat(end_coords, dim=0)

    return num_bins, chrom_offset, chromosomes, start_coords, end_coords

def _initialize_lazy_constants() -> None:
    """Initialize all lazy-loaded constants and assign them to the module."""
    global _initialized
    
    if _initialized:
        return
        
    with _init_lock:
        if _initialized:
            return

        num_bins, chrom_offset, chromosomes, start_coords, end_coords = compute_bins_coordinates(BASE_RESOLUTION)

        # Assign directly to module
        current_module = sys.modules[__name__]
        current_module.num_bins = num_bins
        current_module.chrom_offset = chrom_offset
        current_module.chromosomes = chromosomes
        current_module.start_coords = start_coords
        current_module.end_coords = end_coords
        
        _initialized = True

def __getattr__(name: str):
    """Lazy loading - only called once per attribute."""
    if name in _LAZY_ATTRS:
        _initialize_lazy_constants()
        return getattr(sys.modules[__name__], name)
    else:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")
