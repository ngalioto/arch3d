# ARCH3D
A foundation model for global genome architecture

## Download
The model can be downloaded from [https://huggingface.co/ngalioto/ARCH3D](https://huggingface.co/ngalioto/ARCH3D).

## Install
```bash
python -m pip install --upgrade pip
python -m pip install .
```

## Scripts

<details>
<summary>Generate pre-trained embeddings</summary>

1. Create an mcool file with the desired resolution
2. Perform observed/expected normalization on the Hi-C experiment using [`toeplitz_normalize.py`](digitalcell/data/toeplitz_normalize.py)

**Parameters:**
- `file_name` (str): Path to mcool file
- `save_dir` (str): Directory to save the normalized matrix
- `save_name` (str): Output filename
- `weights_dir` (str, optional): Directory to save pertinent normalization values
- `resolution` (int): Resolution at which to normalize (default: 5000)
- `balance` (bool): If True, normalize balanced Hi-C; otherwise normalize raw Hi-C (default: True)

**Outputs:**
- `{save_dir}/{save_name}.npz` (`scipy.sparse.csr_matrix`): The normalized matrix with chromosomes 1--22

The rest are saved only if the argument `weights_dir` is passed
- `{weights_dir}/{save_name}_pixels.npy` (`numpy.ndarray`): Array holding the sum of all pixels along each diagonal. The tensor index corresponds to the diagonal offset.
- `{weights_dir}/{save_name}_counts.npy` (`numpy.ndarray`): Array holding the number of nonzero pixesl along each diagonal. The tensor index corresponds to the diagonal offset.

**Example:**
```bash
python digitalcell/data/toeplitz_normalize.py \
    "file_name" \
    "save_dir" \
    --save_name "save_name" \
    --weights_dir "weights_dir" \
    --resolution 5000 \
    --balance True
```

3. Generate the embeddings with [`generate_embeddings.py`](digitalcell/scripts/generate_embeddings.py)

**Parameters:**
- `ckpt_path` (str): Path to the HiCT model checkpoint
- `data_file` (str): Path to the HiC data file
- `resolution` (int): Resolution for binning the data
- `save_dir` (str): Directory for saving the embeddings
- `shuffle` (bool, optional): Shuffle loci before generating embeddings (default: True)

**Outputs:**
- `{save_dir}/embeddings.pt` (`torch.Tensor`): The generated embeddings
- `{save_dir}/mappable_idx.pt` (`torch.Tensor`): Boolean mask of indices where the rows are not all zeros
- `{save_dir}/chromosomes.pt` (`torch.Tensor`): Chromosome assignments (zero-indexed)
- `{save_dir}/start_bp.pt` (`torch.Tensor`): Starting base pair positions (zero-indexed and in Mbp)

**Example:**
```bash
python digitalcell/scripts/generate_embeddings.py \
    --ckpt_path "path/to/checkpoint" \
    --data_file "path/to/data.npz" \
    --resolution 5000 \
    --save_dir "embeddings_output" \
    --shuffle True
```

</details>


## Cite
Galioto, Nicholas, et al. "ARCH3D: A foundation model for global genome architecture." _bioRxiv_ (2026): 2026-02.
```bibtex
@article{galioto2026arch3d,
  title={{ARCH3D}: A foundation model for global genome architecture},
  author={Galioto, Nicholas and Stansbury, Cooper and Gorodetsky, Alex Arkady and Rajapakse, Indika},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```