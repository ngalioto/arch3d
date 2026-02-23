import math
import os
import random
from collections.abc import Iterator
from typing import Iterable, TypeVar

import torch
import torch.distributed as dist
import lightning as L
from torch.utils.data import DataLoader, DistributedSampler

from digitalcell.data.dataset import HiC_Dataset, HiC_Sequence, collate_hic_sequences

_T_co = TypeVar("_T_co", covariant=True)


"""
This file contains the dataloader for the Hi-C dataset.

This dataloader is used for loading batches of Hi-C data for training.
Data augmentation methods might also go here, or maybe they go in the training loop.
"""
    

class ShardedSampler(DistributedSampler):
    
    def __init__(
        self,
        dataset: HiC_Dataset,
        total_file_num: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        
        super().__init__(
            dataset = dataset,
            shuffle = shuffle,
            seed = seed,
            drop_last = drop_last
        )
        
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.floor(total_file_num / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_file_num / self.num_replicas) 
        
        self.total_size = self.num_samples * self.num_replicas
        
    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset))) 

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.num_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.num_samples]

        assert len(indices) == self.num_samples

        return iter(indices)


def partition_files_by_number(file_paths, num_partitions):
    
    partitions = [file_paths[ii::num_partitions] for ii in range(num_partitions)]
    
    return partitions


def compile_file_list(
    data_dir: str,
    seed: int = 0,
    blacklist: Iterable[str] = ['']
) -> dict[str, str]:
    
    data_dir = data_dir if isinstance(data_dir, list) else [data_dir]
    all_files = {}
    # First collect all files to process
    for path in data_dir:
        for f in os.listdir(path):
            accession = os.path.splitext(f)[0]
            if accession in blacklist:
                print(f'Skipping {accession} due to blacklist')
                continue
            elif accession in all_files:
                print(f'Found duplicate {accession}. Skipping...')
                continue
            array = os.path.join(path, f)
            
            all_files[accession] = array

    return all_files

def load_prespecified_splits(
    metadata_dir: str | None = None
) -> tuple[list[str], list[str], list[str]]:
    """
    Load prespecified train, val, and test file splits from metadata directory.
    
    Args:
        metadata_dir: Directory containing split files
        
    Returns:
        Tuple of (train_files, val_files, test_files) lists
    """
    train_files = []
    val_files = []
    test_files = []
    
    if metadata_dir is not None:
        # Check for existing splits in subdirectories
        for subdir in os.listdir(metadata_dir):
            if subdir.startswith('splits_rank'):
                subdir_path = os.path.join(metadata_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Load train files
                    train_file_path = os.path.join(subdir_path, 'train.txt')
                    if os.path.exists(train_file_path):
                        with open(train_file_path, 'r') as f:
                            train_files.extend([os.path.splitext(os.path.basename(line.strip()))[0] for line in f.readlines() if line.strip()])
                    
                    # Load val files
                    val_file_path = os.path.join(subdir_path, 'val.txt')
                    if os.path.exists(val_file_path):
                        with open(val_file_path, 'r') as f:
                            val_files.extend([os.path.splitext(os.path.basename(line.strip()))[0] for line in f.readlines() if line.strip()])
                    
                    # Load test files
                    test_file_path = os.path.join(subdir_path, 'test.txt')
                    if os.path.exists(test_file_path):
                        with open(test_file_path, 'r') as f:
                            test_files.extend([os.path.splitext(os.path.basename(line.strip()))[0] for line in f.readlines() if line.strip()])
            else:
                continue
        
        train_files = set(train_files); val_files = set(val_files); test_files = set(test_files)

        val_files = set() # this can be removed after this stage        
        
        print(f"Found {len(train_files)} train, {len(val_files)} val, and {len(test_files)} test accessions from {metadata_dir}")
    
    return train_files, val_files, test_files


class HiC_DataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        data_split: Iterable[float | int] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_workers: int = 10,
        prefetch_factor: int = 2,
        metadata_dir: str | None = None,
        drop_last: bool = True,
        max_files: int | None = None,
        blacklist: Iterable[str] = [''],
        metadata_ckpt: str | None = None,
        dataset_config: dict | None = None
    ) -> None:
        
        """
        """
        
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_split = data_split
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.metadata_dir = metadata_dir
        self.drop_last = drop_last
        self.max_files = max_files
        self.blacklist = blacklist
        self.metadata_ckpt = metadata_ckpt
        
        self.total_train_len = 0
        self.total_val_len = 0
        self.total_test_len = 0
        
        self.shard_data = False
        self.global_rank = 0
        
        self.dataset_config = dataset_config

        self.save_hyperparameters()

    def _create_train_val_test_split(self, world_size: int) -> Iterable:

        file_list = compile_file_list(
            data_dir=self.data_dir,
            seed=self.seed,
            blacklist=self.blacklist
        )

        num_files = len(file_list) if self.max_files is None else min(len(file_list), self.max_files)
        num_val_files = int(num_files * self.data_split[1])
        if self.data_split[2] > 0:
            num_train_files = int(num_files * self.data_split[0])
            num_test_files = num_files - num_train_files - num_val_files
        else:
            num_train_files = num_files - num_val_files
            num_test_files = 0

        # Load prespecified file splits if available
        train_split, val_split, test_split = load_prespecified_splits(self.metadata_ckpt)

        # convert accession numbers to paths, excluding files not in `file_list`
        train_split = [file_list[accession] for accession in train_split if accession in file_list]
        val_split = [file_list[accession] for accession in val_split if accession in file_list]
        test_split = [file_list[accession] for accession in test_split if accession in file_list]

        split_configs = {
            'train': (train_split, num_train_files),
            'val': (val_split, num_val_files),
            'test': (test_split, num_test_files)
        }

        # If the prespecified splits are longer than the target size, trim them
        for split_name, (split_list, target_size) in split_configs.items():
            if len(split_list) > target_size:
                print(f"Trimming {split_name} split from {len(split_list)} to {target_size} files")
                split_list = split_list[:target_size]

        # If the prespecified splits are shorter than the target size, we will fill them with remaining files
        # Get the remaining files that are not in any of the prespecified splits
        remaining_files = [value for value in file_list.values() if value not in train_split and value not in val_split and value not in test_split]
        # Shuffle the remaining files
        random.Random(self.seed).shuffle(remaining_files)

        splitting_index = 0
        for (split_list, target_size) in split_configs.values():
            num_files_to_add = target_size - len(split_list)
            if num_files_to_add > 0:
                split_list.extend(remaining_files[splitting_index:splitting_index + num_files_to_add])
                splitting_index += num_files_to_add

        if self.shard_data:
            train_partition = partition_files_by_number(train_split, world_size)
            val_partition = partition_files_by_number(val_split, world_size)
            test_partition = partition_files_by_number(test_split, world_size)
            file_partition = [train_partition, val_partition, test_partition]
        else:
            file_partition = [train_split, val_split, test_split]

        return file_partition
        
    def _ddp_setup(self, stage: str) -> None:
        
        self.global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f'GLOBAL_RANK: {self.global_rank}')
        
        if self.global_rank == 0:
            file_partition = self._create_train_val_test_split(world_size)

        else:
            file_partition = None
        
        
        obj_list = [file_partition]
        dist.broadcast_object_list(obj_list, src=0)
        file_partition = obj_list[0]

        self.total_files = sum(len(partition) for split in file_partition for partition in split)

        self.train = HiC_Dataset(
            file_partition[0][self.global_rank],
            **self.dataset_config
        )
        self.val = HiC_Dataset(
            file_partition[1][self.global_rank],
            **self.dataset_config
        )
        self.test = HiC_Dataset(
            file_partition[2][self.global_rank],
            **self.dataset_config
        )
        
        local_train_len = len(self.train)
        local_val_len = len(self.val)
        local_test_len = len(self.test)

        # Accumulate across ranks
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_train_len = torch.tensor(local_train_len, device=device)
        self.total_val_len = torch.tensor(local_val_len, device=device)
        self.total_test_len = torch.tensor(local_test_len, device=device)

        dist.all_reduce(self.total_train_len, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_val_len, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_test_len, op=dist.ReduceOp.SUM)
            
        if self.global_rank == 0:
            print(f"Total train files: {self.total_train_len.item()}")
            print(f"Total val files: {self.total_val_len.item()}")
        

    def setup(self, stage: str = "") -> None:

        """
        Here we create the HiC_Dataset object and split it into train, val, and test sets.
        """
        
        
        if dist.is_available() and dist.is_initialized():
            self.shard_data = True
            self._ddp_setup(stage)
        else:
            file_partition = self._create_train_val_test_split(world_size=1)
            
            self.train = HiC_Dataset(
                file_partition[0],
                **self.dataset_config
            )
            self.val = HiC_Dataset(
                file_partition[1],
                **self.dataset_config
            )
            self.test = HiC_Dataset(
                file_partition[2],
                **self.dataset_config
            )

        print('Finishing setup')
        
        if self.metadata_dir is not None:
            # Save file splits
            split_dir = os.path.join(self.metadata_dir, f'splits_rank{self.global_rank}')
            os.makedirs(split_dir, exist_ok=True)

            self.write_split(self.train, os.path.join(split_dir, 'train.txt'))
            self.write_split(self.val, os.path.join(split_dir, 'val.txt'))
            self.write_split(self.test, os.path.join(split_dir, 'test.txt'))
    
    def write_split(
        self, 
        split: HiC_Dataset, 
        split_file: str
    ) -> None:
        
        with open(split_file, "w") as f:
            for data in split.data:
                f.write(f"{data.fname}\n")

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, 'train'):
            raise ValueError('DataModule has not been set up. Call `setup()` before calling this method.')
            
        shuffle = True
            
        sampler = ShardedSampler(
            dataset = self.train,
            total_file_num = self.total_train_len,
            shuffle = shuffle,
            drop_last = self.drop_last
        ) if self.shard_data else None
            
        return DataLoader(
            self.train, 
            batch_size = self.batch_size,
            collate_fn = self.collate_fn,
            num_workers = self.num_workers,
            prefetch_factor = self.prefetch_factor,
            sampler = sampler,
            shuffle = shuffle if not self.shard_data else None
        )

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, 'val'):
            raise ValueError('DataModule has not been set up. Call `setup()` before calling this method.')
            
        shuffle = False
            
        sampler = ShardedSampler(
            dataset = self.val,
            total_file_num = self.total_val_len,
            shuffle = shuffle,
            drop_last = self.drop_last
        ) if self.shard_data else None
        
        return DataLoader(
            self.val, 
            batch_size = self.batch_size,
            collate_fn = self.collate_fn,
            num_workers = self.num_workers,
            prefetch_factor = self.prefetch_factor,
            sampler = sampler,
            shuffle = shuffle if not self.shard_data else None
        )

    def test_dataloader(self)  -> DataLoader:
        if not hasattr(self, 'test'):
            raise ValueError('DataModule has not been set up. Call `setup()` before calling this method.')
            
        shuffle = False
            
        sampler = ShardedSampler(
            dataset = self.test,
            total_file_num = self.total_val_len,
            shuffle = shuffle,
            drop_last = self.drop_last
        ) if self.shard_data else None
        
        return DataLoader(
            self.test, 
            batch_size = self.batch_size,
            collate_fn = self.collate_fn,
            num_workers = self.num_workers,
            prefetch_factor = self.prefetch_factor,
            sampler = sampler,
            shuffle = shuffle if not self.shard_data else None
        )
    
    def collate_fn(self, batch: Iterable[HiC_Sequence]) -> list[HiC_Sequence]:
        return collate_hic_sequences(batch)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...