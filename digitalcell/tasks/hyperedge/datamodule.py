import os
import torch
import pickle
import numpy as np
import lightning as L
from math import isclose
from typing import Iterable
from digitalcell.data import constants
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from digitalcell.tasks.hyperedge.matcha import load_hyperedge_data, generate_negative, build_hash


class Hyperedge_Experiment(Dataset):
    def __init__(
        self,
        clusters: Iterable,
        weights: Iterable,
        embeddings: torch.Tensor,
        min_size: int,
        max_size: int,
        train_dict: dict,
        data_name: str,
        dataset_config: dict,
        correlation_matrix_path: str | None = None,
        seed: int = 42,
        model_flag: str = 'arch3d'
    ) -> None:

        self.clusters = clusters
        self.weights = weights
        self.embeddings = embeddings
        self.train_dict = train_dict
        
        self.neg_num = dataset_config.get('neg_num', 3)
        
        self.min_distance = dataset_config.get('min_distance', 0)
        resolution = dataset_config.get('resolution', 100000)
        self.min_size = min_size
        self.max_size = max_size
        self.data_name = data_name
        self.generator = torch.Generator().manual_seed(seed)
        self.model_flag = model_flag
        
        self.scaling_factor = None
        
        _, chrom_offset, self.node2chrom, _, _ = constants.compute_bins_coordinates(resolution)
        self.chrom_range = torch.stack((chrom_offset[:-1], chrom_offset[1:]), dim=1).numpy()
        
        if correlation_matrix_path is not None:
            self.correlation_matrix = torch.load(correlation_matrix_path, map_location="cpu", weights_only=True)
        elif model_flag == 'multicorr':
            raise ValueError("The `model_flag` in datamodule is set to 'multicorr', but `correlation_matrix_path` was not passed.")
        else:
            self.correlation_matrix = None
        

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Gets the corresponding processed Hi-C and epigenomic data object for the given index `idx`.
        """
        # List of hyperedge positive samples
        hyperedges = np.expand_dims(self.clusters[idx], axis=0)
        weights = self.weights[idx].unsqueeze(0)
        size = len(hyperedges[0])

        # Generate negative samples
        hyperedges, labels, weights, size_list = generate_negative(
            hyperedges,
            data_dict=self.train_dict,
            weight=weights,
            neg_num=self.neg_num,
            min_size=self.min_size,
            max_size=self.max_size,
            min_dis=self.min_distance,
            node2chrom=self.node2chrom,
            chrom_range=self.chrom_range,
            generator=self.generator
        )
        
        if self.scaling_factor is not None:
            weights *= self.scaling_factor
            
        if self.model_flag == "multicorr":
            hyperedge_indices = hyperedges.reshape(-1, size_list[0])
            # shape (neg_num + 1) x size x size
            correlation_matrices = self.correlation_matrix[hyperedge_indices.unsqueeze(-1), hyperedge_indices.unsqueeze(-2)]
            
            return correlation_matrices, labels, weights, size_list, hyperedges, self.data_name
        
        elif self.model_flag == "matcha":
            return None, labels, weights, size_list, hyperedges, self.data_name

        # Convert hyperedge indices into HiC_Sequence
        embeddings = torch.zeros(
            1 + self.neg_num, self.max_size, self.embeddings.shape[-1],
            dtype=torch.float32, 
            device=self.embeddings.device
        )

        # self.embeddings is of shape (num_bins, embedding_dim)
        embeddings[:, :size] = self.embeddings[hyperedges.flatten()].reshape(1 + self.neg_num, -1, self.embeddings.shape[-1])

        # return `hyperedges` for use in statistical baseline methods
        return embeddings.unsqueeze(0), labels, weights, size_list, hyperedges, self.data_name

class HyperedgeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Iterable[str],
        hic_path: Iterable[str],
        embeddings_path: Iterable[str],
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_config: dict = {},
        train_split: Iterable[float | int] | None = None,
        metadata_dir: str | None = None,
        correlation_matrix_path: Iterable[str] | None = None,
        model_flag: str = "arch3d",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # Initialize placeholders
        self.train_ds = None
        self.val_ds = None
        kmer_size = dataset_config.get('k-mer_size', [3, 4, 5])
        self.min_size = int(min(kmer_size))
        self.max_size = int(max(kmer_size))
        self.hash_path = [os.path.join(
            data_dir, 
            f"hash_table_min{self.min_size}_max{self.max_size}.pkl"
        ) for data_dir in self.hparams.data_dir]

    def prepare_data(self):
        """Runs ONLY on Rank 0. Builds hash table if not exists."""
        for hash_path, data_dir in zip(self.hash_path, self.hparams.data_dir):
            if not os.path.exists(hash_path):
                print("Building hash table on Rank 0...")
                clusters, _ = load_hyperedge_data(
                    data_dir,
                    self.hparams.dataset_config['k-mer_size'],
                    quantile_cutoff_for_unlabel=self.hparams.dataset_config.get('quantile_cutoff_for_unlabel', 0.4),
                    neg_num=self.hparams.dataset_config['neg_num']
                ) # Minimal load to build hash
                train_dict = build_hash(clusters, self.min_size, self.max_size)
                with open(hash_path, "wb") as f:
                    pickle.dump(train_dict, f)
                    

    def setup(self, stage: str | None = None):
        """Runs on EVERY GPU independently."""
        train_datasets = []
        val_datasets = []
        dataset_size = []

        # 1. Load Data (Every GPU gets a copy in RAM)
        for dataset_idx, (data_dir, embeddings_path, hash_path) in enumerate(zip(self.hparams.data_dir, self.hparams.embeddings_path, self.hash_path)):
            
            if self.hparams.correlation_matrix_path is not None:
                correlation_matrix_path = self.hparams.correlation_matrix_path[dataset_idx]
            else:
                correlation_matrix_path = None

            data_name = os.path.basename(data_dir.rstrip('/'))

            clusters, weights = load_hyperedge_data(
                data_dir,
                self.hparams.dataset_config['k-mer_size'],
                quantile_cutoff_for_unlabel=self.hparams.dataset_config.get('quantile_cutoff_for_unlabel', 0.4),
                neg_num=self.hparams.dataset_config['neg_num']
            )
            
            # Load embeddings (Weights only = True is good for safety)
            embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=True)
            # 2. Load the pre-computed hash table
            with open(hash_path, "rb") as f:
                train_dict = pickle.load(f)
    
            full_ds = Hyperedge_Experiment(
                clusters=clusters,
                weights=weights,
                embeddings=embeddings,
                min_size=self.min_size,
                max_size=self.max_size,
                train_dict=train_dict,
                data_name=data_name,
                dataset_config=self.hparams.dataset_config,
                seed=self.hparams.seed + dataset_idx,
                correlation_matrix_path=correlation_matrix_path,
                model_flag=self.hparams.model_flag
            )
            dataset_size.append(len(full_ds))

            # --- train/val split ---
            split_fracs = self.hparams.train_split
            if split_fracs is None:
                # if user doesn't want val split, just train on all data
                train_datasets.append(full_ds)
                val_datasets.append(None)
            else:
                try:
                    split_vals = list(split_fracs)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"train_split must be an iterable of two numbers: (train, val). Got {split_fracs}"
                    )

                if len(split_vals) > 2:
                    raise ValueError(
                        f"train_split must cannot contain more than two values: (train, val). Got {split_fracs}"
                    )
                n_total = len(full_ds)
                train_split_val = float(split_vals[0])
                val_split_val = float(split_vals[1]) if len(split_vals) == 2 else -1
                both_int_like = train_split_val.is_integer() and val_split_val.is_integer()

                if 0.0 <= train_split_val < 1.0 and (0.0 <= val_split_val < 1.0 or val_split_val == -1):
                    # Fraction mode: values represent percentages of the dataset.
                    if val_split_val == -1:
                        val_split_val = 1.0 - train_split_val
                    elif train_split_val + val_split_val > 1.0:
                        raise ValueError(
                            "train_split fractions must satisfy train + val <= 1. "
                            f"Got train={train_split_val}, val={val_split_val}"
                        )

                    n_train = int(n_total * train_split_val)
                    if isclose(train_split_val + val_split_val, 1.0, rel_tol=0.0, abs_tol=1e-9):
                        n_val = n_total - n_train  # assign all remaining samples to val
                    else:
                        n_val = int(n_total * val_split_val)
                elif both_int_like and train_split_val >= 0.0 and val_split_val >= 0.0:
                    # Count mode: values represent explicit sample counts.
                    n_train = int(train_split_val)
                    n_val = int(val_split_val) if val_split_val != -1 else n_total - n_train
                    if n_train + n_val > n_total:
                        raise ValueError(
                            "train_split sample counts must satisfy train + val <= dataset size. "
                            f"Got train={n_train}, val={n_val}, dataset_size={n_total}"
                        )
                else:
                    raise ValueError(
                        "train_split must be either fractions in [0,1) "
                        "or non-negative integer sample counts. "
                        f"Got train={split_vals[0]}, val={split_vals[1]}"
                    )

                n_unused = n_total - n_train - n_val

                g = torch.Generator().manual_seed(int(self.hparams.seed) + dataset_idx)
                train_ds, val_ds, _ = random_split(full_ds, [n_train, n_val, n_unused], generator=g)
                train_datasets.append(train_ds)
                val_datasets.append(val_ds if n_val > 0 else None)
                
        # filter validation
        val_datasets_filtered = [ds for ds in val_datasets if ds is not None]
        
        # Concatenate all train/val datasets
        self.train_ds = ConcatDataset(train_datasets)
        if val_datasets_filtered:
            self.val_ds = ConcatDataset(val_datasets_filtered)
        else:
            self.val_ds = None

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        if self.val_ds is None:
            return None
        else:
            return DataLoader(
                self.val_ds,
                batch_size=self.hparams.batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.hparams.num_workers,
                shuffle=False
            )
        
    def predict_dataloader(self):
        """Use validation set for predictions."""
        if self.val_ds is None:
            raise ValueError("No validation set available.")
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )
    
    def collate_fn(
        self, 
        batch: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[str]]:
    
        if self.hparams.model_flag == "arch3d":
            embeddings = torch.cat([item[0] for item in batch], dim=0)
        elif self.hparams.model_flag == "multicorr":
            embeddings = [item[0] for item in batch]
        elif self.hparams.model_flag == "matcha":
            embeddings = None
            
        label_batch = torch.cat([item[1] for item in batch], dim=0)
        weight_batch = torch.cat([item[2] for item in batch], dim=0)
        size_list = torch.stack([item[3] for item in batch], dim=0)
        hyperedges = [item[4] for item in batch]
        data_names = [item[5] for item in batch]

        return embeddings, label_batch, weight_batch, size_list, hyperedges, data_names

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...
