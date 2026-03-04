import argparse
import math
import os

import torch

from digitalcell.data.constants import BASE_RESOLUTION
from digitalcell.data.dataset import HiC_Data, get_resolution_map, tokenize_data
from digitalcell.model.hict import HiCT

def generate_indices(
    num_loci, 
    num_sequences, 
    seq_length, 
    shuffle=True, 
    seed=42
) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    all_indices = torch.randperm(num_loci, dtype=torch.int64) if shuffle else torch.arange(num_loci, dtype=torch.int64)

    remainder = num_loci - num_sequences * seq_length
    q, r = divmod(remainder, num_sequences)

    idx = []
    cur = 0
    for ii in range(num_sequences):
        length = seq_length + q + (ii < r)
        idx.append(all_indices[cur:cur + length])
        cur += length

    assert cur == num_loci
    return idx

def generate_embeddings(
    model: HiCT,
    data: HiC_Data,
    resolution: int,
    seq_length: int,
    shuffle: bool = True,
    seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    start, stop = get_resolution_map(resolution // BASE_RESOLUTION)
    
    num_loci = len(start)
    model_dim = model.config.d_model
    num_sequences = max(1, math.floor(num_loci / seq_length))

    all_indices = generate_indices(num_loci, num_sequences, seq_length, shuffle, seed)

    # Pre-allocate tensors for results
    embeddings = torch.zeros(num_loci, model_dim, device="cpu")
    mappable_idx = torch.zeros(num_loci, dtype=torch.bool, device="cpu")
    start_bp = torch.zeros(num_loci, device="cpu")
    chromosomes = torch.zeros(num_loci, dtype=torch.int32, device="cpu")

    model.to(device)

    with torch.no_grad():
        for idx in all_indices:
            hic_seq = tokenize_data(data, start[idx], stop[idx])
            mappable_idx[idx] = (torch.sum(hic_seq.input_loci.values[0], dim=-1) > 0).flatten().cpu()
            chromosomes[idx] = hic_seq.input_loci.chromosomes.flatten().cpu()
            start_bp[idx] = hic_seq.input_loci.start.flatten().cpu()
            embeddings[idx] = model(hic_seq.to(device)).cpu().squeeze()
    del hic_seq

    return embeddings, mappable_idx, chromosomes, start_bp

def embed_experiment(
    ckpt_path: str,
    data_file: str,
    resolution: int,
    shuffle: bool = True
):
    
    print(f'Loading data from {data_file}...')
    data = HiC_Data(data_file)

    model = HiCT.load_from_checkpoint(ckpt_path)
    model.eval()

    return generate_embeddings(
        model,
        data,
        resolution,
        seq_length=1024,
        shuffle=shuffle
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate embeddings using HiCT model.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the HiCT model checkpoint.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the HiC data file.')
    parser.add_argument('--resolution', type=int, required=True, help='Resolution for binning the data.')
    parser.add_argument('--save_dir', type=str, required=True, help='File path for saving the embeddings')
    parser.add_argument('--shuffle', type=bool, required=False, default=True, help='Shuffle loci or not before generating embeddings')

    args = parser.parse_args()

    embeddings, mappable_idx, chromosomes, start_bp = embed_experiment(
        ckpt_path=args.ckpt_path,
        data_file=args.data_file,
        resolution=args.resolution,
        shuffle=args.shuffle,
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(embeddings, os.path.join(args.save_dir, 'embeddings.pt'))
    torch.save(mappable_idx, os.path.join(args.save_dir, 'mappable_idx.pt'))
    torch.save(chromosomes, os.path.join(args.save_dir, 'chromosomes.pt'))
    torch.save(start_bp, os.path.join(args.save_dir, 'start_bp.pt'))

    print("Embeddings generation completed.")