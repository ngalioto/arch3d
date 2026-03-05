import torch
import math
import numpy as np
import random
import json
import os
from tqdm import tqdm
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from pybloom_live import BloomFilter
from sklearn.preprocessing import QuantileTransformer

from digitalcell.data import constants


"""
The code in this file is adapted from the open-source repository MATCHA
The source code can be found at https://github.com/ma-compbio/MATCHA
The associated publication can be found at https://www.cell.com/cell-systems/fulltext/S2405-4712(20)30147-2
"""

def build_hash(data, min_size, max_size, capacity=None) -> list:

    if capacity is None:
        capacity = len(data) * 5
        capacity = math.ceil(capacity) + 1000
        print("total_capacity", capacity)
    dict_list = []
    for i in range(max_size + 1):
        if i < min_size:
            dict_list.append(BloomFilter(10, 1e-3))
        else:
            dict_list.append(BloomFilter(capacity, 1e-3))
    
    print(f'{min_size=}, {max_size=}')

    for datum in tqdm(data, desc="Building hash table"):
        dict_list[len(datum)].add(tuple(datum))

    print(f'Capacity used: {len(dict_list[min_size]) / dict_list[min_size].capacity}')

    print(f'Total number of hyperedges: {len(dict_list[-1])}')
    length_list = [len(dict_list[i]) for i in range(len(dict_list))]
    print(f'Total number of hyperedges at every size: {length_list}')

    return dict_list

def quantile_ties_up(x: np.ndarray) -> np.ndarray:
    n = x.size
    xs = np.sort(x)
    # for each x_i, count how many values are <= x_i
    counts_le = np.searchsorted(xs, x, side="right")
    return counts_le / n

def load_hyperedge_data(
    temp_dir, 
    size_list,
    quantile_cutoff_for_unlabel: float = 0.4,
    neg_num: int = 3
):

    """
    Load hyperedge data and corresponding weights from temporary directory. Any hyperedges with frequency quantiles that fall below `quantile_cutoff_for_unlabel` are filtered out.

    Parameters:
    ----------
    temp_dir: str
        Path to the temporary directory containing hyperedge data files.
    size_list: List[int]
        List of hyperedge sizes to load.
    quantile_cutoff_for_unlabel: float
        Quantile cutoff for filtering hyperedges based on their weights.

    Returns:
    -------
    data_list: List[np.ndarray]
        List of hyperedges after filtering. 
        Length of list is number of hyperedges. 
        Size of each hyperedge is the hyperedge order.
        We keep this as a list of numpy arrays, not tensors, for compatibility with Bloom filter.
    weight: torch.Tensor
        Corresponding weights for the filtered hyperedges. Length of array is number of hyperedges.
    """
    
    data_list = []
    weight_list = []

    for size in size_list:
        data = np.load(os.path.join(temp_dir,"all_%d_counter.npy" % size)).astype('int')
        weight = np.load(os.path.join(temp_dir,"all_%d_freq_counter.npy" % size)).astype('float32')
        print("before filter", "size", size, "length", len(data))
        weight = quantile_ties_up(weight)
        mask = weight > quantile_cutoff_for_unlabel
        data = data[mask]
        weight = weight[mask]
        print("after filter", "size", size, "length", len(data))
        for datum in data:
            data_list.append(datum)
        weight_list.append(weight)

    weight = torch.from_numpy(np.concatenate(weight_list,axis = 0))

    weight /= weight.mean()
    weight *= neg_num

    return data_list, weight

def np2tensor_hyper(vec, dtype):
	vec = np.asarray(vec)
	if len(vec.shape) == 1:
		return [torch.as_tensor(v, dtype=dtype) for v in vec]
	else:
		return torch.as_tensor(vec, dtype=dtype)

def generate_negative(
    hyperedges: Iterable[np.ndarray], 
    data_dict: str, 
    weight: torch.Tensor | None = None, 
    neg_num: int = 3,
    min_size: int = 0,
    max_size: int = 25,
    min_dis: int = 0,
    node2chrom: np.ndarray | None = None,
    chrom_range: np.ndarray | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    """
    Generate negative samples for hyperedges by randomly altering nodes in the hyperedge.

    Parameters:
    ---------
    x: List of hyperedges (each hyperedge is a list of node indices)
    dict1: str, either 'train_dict' or 'test_dict' indicating which dictionary
            to use for checking existing hyperedges
    weight: List of weights corresponding to each hyperedge in x. If empty, all weights are set to 1.
    neg_num: Number of negative samples to generate per positive sample

    Returns:
    -------
    x: Tensor of shape (num_samples, max_hyperedge_size) containing both positive
        and negative samples, padded with zeros where necessary
    y: Tensor of shape (num_samples, 1) containing labels (1 for positive, 0 for negative)
    w: Tensor of shape (num_samples, 1) containing weights for each sample
    size_list: Tensor containing the sizes of each hyperedge (both positive and negative)
    """

    num_hyperedges = len(hyperedges)

    if weight is None: # use uniform weights by default
        weight = torch.ones(len(hyperedges), dtype=torch.float)


    # list of `max_size` empty lists to store number of changes for each hyperedge size
    change_num_list: list[list[int]] = [[] for _ in range(max_size + 1)]
    for size in range(min_size, max_size + 1):
        change_num = []
        while len(change_num) < num_hyperedges * neg_num: # ensure enough change numbers are generated
            # change each hyperedge node with probability 0.5
            change_num = np.random.binomial(size, 0.5, num_hyperedges * neg_num * 2)
            # only consider hyperedges with at least one change
            # As long as over half are non-zero, we will have enough
            change_num = change_num[change_num != 0]

        # store the change numbers for this hyperedge size
        change_num_list[size] = list(change_num)

    neg_list: List[List[int]] = []

    
    neg_weight: List[float] = []
    pos_sizes: List[int] = []
    neg_sizes: List[int] = []

    for j, pos_sample in enumerate(hyperedges):
        hyperedge_size = len(pos_sample)

        for i in range(neg_num):
            decompose_sample = np.copy(pos_sample)
            list1 = change_num_list[hyperedge_size]
            change_num = list1.pop() # number of nodes to change
            nodes_to_change = torch.randperm(hyperedge_size, generator=generator)[:change_num]

            edge = np.copy(decompose_sample)

            # generate samples until one that is not in the dictionary is found
            while tuple(edge) in data_dict[hyperedge_size]:
                edge = np.copy(decompose_sample)

                for node in nodes_to_change:
                    chrom = node2chrom[edge[node]]
                    start, end = chrom_range[chrom]

                    # randomly choose a bin on the chromosome
                    edge[node] = math.floor((end-start) * random.random()) + start


                edge = sorted(list(set(edge)))

                # Ensure the generated hyperedge has the correct size
                if len(edge) < hyperedge_size:
                    edge = np.copy(decompose_sample)
                    continue

                # Ensure minimum distance constraint is satisfied for all nodes in `edge`
                dis_list = []
                for k in range(hyperedge_size - 1):
                    dis_list.append(edge[k + 1] - edge[k])
                if min(dis_list) <= min_dis:
                    edge = np.copy(decompose_sample)
                    continue

            if i == 0:
                pos_sizes.append(hyperedge_size)
            # Add negative sample to the list
            if len(edge) > 0:
                neg_list.append(edge)
                neg_sizes.append(hyperedge_size)
                neg_weight.append(weight[j])

    pos_weight = weight.clone()
    pos_sizes = torch.tensor(pos_sizes + neg_sizes)
    pos_part = np2tensor_hyper(list(hyperedges), dtype=torch.long)
    neg = np2tensor_hyper(neg_list, dtype=torch.long)
    if type(pos_part) == list:
        pos_part = pad_sequence(pos_part, batch_first=True, padding_value=0)
        neg = pad_sequence(neg, batch_first=True, padding_value=0)

    if len(neg) == 0:
        neg = torch.zeros((1, pos_part.shape[-1]),dtype=torch.long, device="cpu")
    pos_part = pos_part
    neg = neg

    y = torch.cat([torch.ones((1, len(pos_part)), device="cpu"),
                    torch.zeros((1, len(neg)), device="cpu")], dim=1)
    w = torch.cat([torch.ones((1, len(pos_part)), device="cpu") * pos_weight.view(1, -1),
                    torch.ones((1, len(neg)), device="cpu")], dim=1)
    hyperedges = torch.cat([pos_part, neg])

    return hyperedges, y, w, pos_sizes