import os
from typing import Iterable

import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from digitalcell.data.datamodule import ShardedSampler
from digitalcell.data.dataset import HiC_Sequence, collate_hic_sequences
from digitalcell.tasks.enhancement.dataset import Enhance_HiC_Dataset

class Enhance_DataModule(L.LightningDataModule):

    def __init__(
        self,
        inputs_dir: str | Iterable[str],
        targets_dir: str | Iterable[str],
        batch_size: int = 1,
        data_split: Iterable[float | int] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_workers: int = 10,
        prefetch_factor: int = 2,
        metadata_dir: str | None = None,
        drop_last: bool = True,
        max_files: int | None = None,
        dataset_config: dict | None = None
    ) -> None:
        
        """
        """
        
        super().__init__()
        
        self.inputs_dir = inputs_dir
        self.targets_dir = targets_dir
        self.batch_size = batch_size
        self.data_split = data_split
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.metadata_dir = metadata_dir
        self.drop_last = drop_last
        self.max_files = max_files
        
        self.total_train_len = 0
        self.total_val_len = 0
        self.total_test_len = 0
        
        self.shard_data = False
        self.global_rank = 0
        
        self.dataset_config = dataset_config

        self.save_hyperparameters()


    def _normalize_data_split(self, data_split, total_samples):
        """
        Convert data split to actual counts.
        
        Args:
            data_split: Tuple of (train, val, test) as either fractions or counts
            total_samples: Total number of samples available
        
        Returns:
            Tuple of (train_count, val_count, test_count) as integers
        """
        train, val, test = data_split
        
        # Check if all values are fractions (between 0 and 1)
        if any(0 < x < 1 for x in [train, val, test]):
            # Ensure they sum to 1 (or close to it)
            total = train + val + test
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Fractions must sum to 1, got {total}")
            
            # Convert to counts
            val_count = int(val * total_samples)
            if test == 0:
                train_count = total_samples - val_count
                test_count = 0
            else:
                test_count = int(test * total_samples)
                train_count = total_samples - val_count - test_count
            
        else:
            # Assume they're already counts
            train_count = int(train)
            val_count = int(val)
            test_count = int(test)
            
            # Verify they don't exceed total
            if train_count + val_count + test_count > total_samples:
                raise ValueError(f"Split counts ({train_count}+{val_count}+{test_count}) exceed total samples ({total_samples})")
        
        return train_count, val_count, test_count

    def _create_train_val_test_split(self, world_size: int) -> Iterable:

        # file list is tuple (hc_data_path, lc_data_path)

        num_files = len(self.inputs_dir) if self.max_files is None else min(len(self.inputs_dir), self.max_files)
        num_train_files, num_val_files, num_test_files = self._normalize_data_split(self.data_split, num_files)

        train_split = (
            self.inputs_dir[:num_train_files],
            self.targets_dir[:num_train_files]
        )
        val_split = (
            self.inputs_dir[num_train_files:num_train_files + num_val_files],
            self.targets_dir[num_train_files:num_train_files + num_val_files]
        ) if num_val_files > 0 else ([], [])
        test_split = (
            self.inputs_dir[-num_test_files:],
            self.targets_dir[-num_test_files:]
        ) if num_test_files > 0 else ([], [])
        
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

        self.train = Enhance_HiC_Dataset(
            *file_partition[0][self.global_rank],
            **self.dataset_config
        )
        self.val = Enhance_HiC_Dataset(
            *file_partition[1][self.global_rank],
            **self.dataset_config
        )
        self.test = Enhance_HiC_Dataset(
            *file_partition[2][self.global_rank],
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
            
            print(f'Train: {file_partition[0]}')
            print(f'Val: {file_partition[1]}')
            
            self.train = Enhance_HiC_Dataset(
                *file_partition[0],
                **self.dataset_config
            )
            self.val = Enhance_HiC_Dataset(
                *file_partition[1],
                **self.dataset_config
            )
            self.test = Enhance_HiC_Dataset(
                *file_partition[2],
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
        split: Enhance_HiC_Dataset, 
        split_file: str
    ) -> None:
        
        with open(split_file, "w") as f:
            f.write(f"Resolutions: {self.dataset_config['resolutions']}\n")
            for inputs in split.data:
                for data in inputs:
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
    
    def collate_fn(self, batch: Iterable[HiC_Sequence]) -> tuple[HiC_Sequence, torch.Tensor]:
        collated_hic_seq = collate_hic_sequences([item[0] for item in batch])
        collated_targets = torch.cat([item[1] for item in batch], dim=0)
        return collated_hic_seq, collated_targets

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...

    def teardown(self, stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...