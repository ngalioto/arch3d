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

<a id="normalize"></a>
<details>
<summary>toeplitz_normalize.py</summary>

Perform observed/expected normalization on a cooler file with [`toeplitz_normalize.py`](digitalcell/data/toeplitz_normalize.py)

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

</details>

<a id="pretrain"></a>
<details>
<summary>pretrain.py</summary>

Pre-train ARCH3D from scratch with [`pretrain.py`](digitalcell/scripts/pretrain.py)

**Parameters:**
- `config` (str): Path to the configuration file

**Example:**
```bash
python digitalcell/scripts/pretrain.py \
    --config "/path/to/config"
```

</details>

<a id="generate-embeddings"></a>
<details>
<summary>generate_embeddings.py</summary>

The steps to generate pre-trained embeddings from a Hi-C experiment are the following:

1. Create an mcool file with the desired resolution
2. Perform observed/expected normalization on the Hi-C experiment using [`toeplitz_normalize.py`](digitalcell/data/toeplitz_normalize.py)
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

## Downstream tasks

### Resolution enhancement



### Hyperedge prediction

The workflow for this task is as follows:

1. Run [`process_clusters.py`](digitalcell/tasks/hyperedge/process_clusters.py)
2. Set the parameters in [`generate_kmers.py`](digitalcell/tasks/hyperedge/generate_kmers.py) and run
3. Generate embeddings from virtual or real Hi-C with [`generate_embeddings.py`](digitalcell/scripts/generate_embeddings.py)
4. Train the model [`hyperedge.py`](digitalcell/tasks/hyperedge/hyperedge.py)
5. Test the predictions with [`test_hyperedge.py`](digitalcell/tasks/hyperedge/test_hyperedge.py)


Each script is explained below:

<details>
<summary>process_clusters.py</summary>

**Parameters:**
- `parent_dir` (str): Path to directory containing cluster files
- `parent_save_dir` (str): Path to directory for saving processed files
- `resolution` (int): Resolution for inference (default: 100000)

**Outputs:**
- `{parent_save_dir}/{parent_dir_name}/edge_list.npy` (`numpy.ndarray`): Hyperedges for inference
- `{parent_save_dir}/{parent_dir_name}/matrix.txt` (tab-separated): Contact matrix pixels
- `{parent_save_dir}/{parent_dir_name}/bins.bed` (BED format): Genomic bins
- `{parent_save_dir}/{parent_dir_name}/output.cool` (cool file): Cooler file at 1 kb resolution
- `{parent_save_dir}/{parent_dir_name}/output.mcool` (mcool file): Multi-resolution Cooler file with balanced Hi-C

**Example:**
```bash
python digitalcell/tasks/hyperedge/process_clusters.py \
  --parent-dir "path/to/directory/containing/clusters/files" \
  --parent-save-dir "path/to/output" \
  --resolution 100000
```

</details>

<details>
<summary>generate_kmers.py</summary>

Generate k-mer hyperedges from `edge_list.npy` with [`generate_kmers.py`](digitalcell/tasks/hyperedge/generate_kmers.py)

**Parameters:**
- `max_cluster_size` (int): Maximum cluster size to consider
- `k_list` (List[int]): List of k-mer sizes to generate
- `temp_dir` (str): Directory containing edge_list.npy and for saving outputs
- `min_freq_cutoff` (int): Minimum frequency threshold for k-mers (default: 2)
- `resolution` (int): Resolution for inference (default: 100000)

**Outputs:**
- `{temp_dir}/all_{k}_counter.npy` (`numpy.ndarray`): k-mer hyperedges for each k in k_list
- `{temp_dir}/all_{k}_freq_counter.npy` (`numpy.ndarray`): Frequency counts for each k-mer

**Example:**
```bash
python digitalcell/tasks/hyperedge/generate_kmers.py \
  --max_cluster_size 25 \
  --k_list [3,4,5] \
  --temp_dir "path/to/temp" \
  --min_freq_cutoff 2 \
  --resolution 100000
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