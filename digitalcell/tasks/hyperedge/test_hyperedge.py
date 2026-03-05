import argparse
import numpy as np
import math
import os
import yaml
import pickle
from pathlib import Path
import lightning as L
from torch.nn.utils.rnn import pad_sequence

from typing import Iterable

import torch
import glob
import re

from digitalcell.conf.utils import load_config
from digitalcell.tasks.hyperedge.datamodule_gpu import HyperedgeDataModule
from digitalcell.tasks.hyperedge.model_gpu import HyperedgeModel

from joblib import Parallel, delayed
import Modules as legacy_modules

class MultiCorr(L.LightningModule):
        
    def predict_step(
        self,
        batch,
        batch_idx: int
    ) -> torch.Tensor:
        
        correlation_matrices, labels, weights, size_list, hyperedges, data_names = batch
        
        batch_size = len(correlation_matrices)
        num_samples = correlation_matrices[0].shape[0]

        vals = torch.zeros(batch_size, num_samples, 3, device=correlation_matrices[0].device, dtype=correlation_matrices[0].dtype)
        for ii, correlation_matrix in enumerate(correlation_matrices):
            evals = torch.linalg.eigvalsh(correlation_matrix)
            vals[ii, :, 0] = 1 - torch.min(evals, dim=-1).values
            vals[ii, :, 1] = torch.sqrt(torch.clamp(1 - torch.linalg.det(correlation_matrix), min=0))
            vals[ii, :, 2] = torch.std(evals, dim=-1, correction=0) / math.sqrt(evals.shape[-1])
        
        return {
            'probs': vals,
            'labels': labels,
            'weights': weights,
            'hyperedges': hyperedges,
            'size_list': size_list,
            'data_names': data_names,
            'batch_idx': batch_idx
        }
    

class MATCHA(L.LightningModule):
    
    def __init__(
        self,
        ckpt_map: dict[str, str] | None = None
    ):
        
        super().__init__()
        
        legacy_modules.device = torch.device("cpu")
        self._active_device = torch.device("cpu")
        self._model_device = torch.device("cpu")
        parent_dir = "/path/to/pretrained/matcha/models"  # Update this path to where the MATCHA models are stored
        data_names = ["GM12878_deshpande_2022", "BJ_fibroblast", "IR_fibroblast"]
        self.models = {
            data_names[ii]: torch.load(os.path.join(parent_dir, f"{data_names[ii]}/matcha/model2load"), map_location="cpu", weights_only=False).eval() \
                for ii in range(len(data_names))
        }

    def _set_models_device(self, device: torch.device) -> None:
        # Legacy MATCHA models in Modules.py store tensors in plain Python
        # attributes/lists that are not consistently moved by .to(device).
        # Keep model execution on CPU to avoid device mismatch at indexing.
        self._active_device = device

    def on_predict_start(self) -> None:
        trainer_device = self.trainer.strategy.root_device
        self._set_models_device(trainer_device)

    @staticmethod
    def _pad_hyperedges(he_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        if not he_list:
            raise RuntimeError("Empty hyperedge list in batch.")

        ndim = he_list[0].dim()
        if any(h.dim() != ndim for h in he_list):
            dims = [h.dim() for h in he_list]
            raise RuntimeError(f"Inconsistent hyperedge ranks in batch: {dims}")

        if ndim == 1:
            return pad_sequence(he_list, batch_first=True, padding_value=0).to(device)
        if ndim == 2:
            max_rows = max(h.shape[0] for h in he_list)
            max_cols = max(h.shape[1] for h in he_list)
            out = torch.zeros(
                (len(he_list), max_rows, max_cols),
                dtype=he_list[0].dtype,
                device=device,
            )
            for i, h in enumerate(he_list):
                out[i, : h.shape[0], : h.shape[1]] = h.to(device)
            return out
        raise RuntimeError(f"Unsupported hyperedge rank {ndim}; expected 1D or 2D tensors.")
        
    def predict_step(
        self,
        batch,
        batch_idx: int
    ) -> torch.Tensor:
        
        _, labels, weights, size_list, hyperedges, data_names = batch
        
        runtime_device = labels.device if torch.is_tensor(labels) else self._active_device
        self._set_models_device(runtime_device)
        probs = torch.empty_like(labels, dtype=torch.float32, device=runtime_device)
        
        for name in set(data_names):
            idx = [i for i, x in enumerate(data_names) if x == name]
            model = self.models[name]

            he = [hyperedges[ii] for ii in idx]
            he = self._pad_hyperedges(he, self._model_device)

            if he.dim() == 3:
                b, s, k = he.shape
                pred = model(he.reshape(b * s, k))
                if pred.dim() > 1 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                pred = pred.reshape(b, s)
            else:
                pred = model(he)
                if pred.dim() > 1 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)

            probs[idx] = pred.to(runtime_device)
        
        return {
            'probs': probs,
            'labels': labels,
            'weights': weights,
            'hyperedges': hyperedges,
            'size_list': size_list,
            'data_names': data_names,
            'batch_idx': batch_idx
        }


def collate_preds(
    data_dir: str,
    batch_size: int,
    num_samples: int
) -> None:
    
    parent_dir = os.path.dirname(data_dir)
    files = sorted(glob.glob(f"{data_dir}/pred_rank*_batch*.pt"))
    
    rank_re = re.compile(r"pred_rank(\d+)_batch(\d+)\.pt")

    max_rank = -1
    max_batch = -1

    for f in files:
        m = rank_re.search(f)
        if m:
            rank = int(m.group(1))
            batch = int(m.group(2))
            max_rank = max(max_rank, rank)
            max_batch = max(max_batch, batch)
            
    total_num = (max_rank + 1) * (max_batch + 1) * batch_size
    
    results = torch.load(f, weights_only=True, map_location="cpu")
    
    extra_dims = results['probs'].shape[2:]  # e.g. (3,)
    probs = torch.zeros(total_num, num_samples, *extra_dims)
    
    labels = torch.zeros(total_num, num_samples)
    weights = torch.zeros(total_num, num_samples)
    sizes = torch.zeros(total_num, num_samples)
    
    data_names = []
    full_size = 0
    global_idx = 0
    for f in files:
        results = torch.load(f, weights_only=True, map_location="cpu")

        this_batch_size = len(results['labels'])
        ending_idx = global_idx + this_batch_size
        full_size = max(full_size, ending_idx)
        global_slice = slice(global_idx, ending_idx)

        probs[global_slice] = results['probs']
        labels[global_slice] = results['labels']
        weights[global_slice] = results['weights']
        sizes[global_slice] = results['size_list']
        data_names.extend(results['data_names'])

        global_idx += this_batch_size
        
    torch.save(probs[:full_size], f"{parent_dir}/probs.pt")
    torch.save(labels[:full_size], f"{parent_dir}/labels.pt")
    torch.save(weights[:full_size], f"{parent_dir}/weights.pt")
    torch.save(sizes[:full_size], f"{parent_dir}/sizes.pt")
    torch.save(data_names[:full_size], f"{parent_dir}/data_names.pt")

    
class PredictWriter(L.Callback):
    def __init__(self, out_dir: str, every_n_batches: int = 1):
        self.out_dir = out_dir
        self.every_n_batches = every_n_batches
        os.makedirs(out_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        if (batch_idx % self.every_n_batches) != 0:
            return

        # Move to CPU + drop graph references
        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            if isinstance(x, list):
                return [to_cpu(v) for v in x]
            if isinstance(x, tuple):
                return tuple(to_cpu(v) for v in x)
            return x

        out = {k: to_cpu(v) for k, v in outputs.items()}

        # Store batch shard
        shard_path = os.path.join(self.out_dir, f"pred_rank{trainer.global_rank}_batch{batch_idx:06d}.pt")
        torch.save(out, shard_path)

def main(
    config: dict = None
) -> None:

    test_multicorr = config.get('test_multicorr', False)
    test_arch3d = config.get('test_arch3d', False)
    test_matcha = config.get('test_matcha', False)
    
    ############ MULTICORR ###########
    if test_multicorr:
        datamodule = HyperedgeDataModule(**config['datamodule'], dataset_config=config['dataset'], model_flag="multicorr")
        datamodule.setup(stage="predict")

        save_dir = config['datamodule']['mc_save_dir']
        out_dir = os.path.join(save_dir, "pred_shards")
        writer = PredictWriter(out_dir)
        model = MultiCorr()

        trainer = L.Trainer(**config['trainer'], callbacks=[writer])
        trainer.predict(model, datamodule=datamodule, return_predictions=False)
        collate_preds(
            data_dir=out_dir,
            batch_size=config['datamodule']['batch_size'],
            num_samples=config['dataset']['neg_num'] + 1
        )

        del datamodule
        torch.cuda.empty_cache()

    ########### ARCH3D ###########
    if test_arch3d:
        # re-initialize datamodule
        config['datamodule']['correlation_matrix_path'] = None
        datamodule = HyperedgeDataModule(**config['datamodule'], dataset_config=config['dataset'], model_flag="arch3d")
        datamodule.setup(stage="predict")

        ckpt_path = config['ckpt_path']
        model = HyperedgeModel.load_from_checkpoint(ckpt_path, map_location="cpu")
        model.eval()

        save_dir = config['datamodule']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        out_dir = os.path.join(save_dir, "pred_shards")
        writer = PredictWriter(out_dir)

        trainer = L.Trainer(**config['trainer'], callbacks=[writer])
        trainer.predict(model, datamodule=datamodule, return_predictions=False)
        collate_preds(
            data_dir=out_dir,
            batch_size=config['datamodule']['batch_size'],
            num_samples=config['dataset']['neg_num'] + 1
        )

        del datamodule
        torch.cuda.empty_cache()

    ######### MATCHA #########
    if test_matcha:
        # re-initialize datamodule
        config['datamodule']['correlation_matrix_path'] = None
        datamodule = HyperedgeDataModule(**config['datamodule'], dataset_config=config['dataset'], model_flag="matcha")
        datamodule.setup(stage="predict")

        ckpt_path = config['ckpt_path']
        # NOTE: This line requires the script to be run from the MATCHA repo root with the legacy Modules.py available. If running from a different context, ensure Modules.py is accessible or adjust the loading mechanism accordingly.
        model = MATCHA()

        save_dir = config['datamodule']['matcha_save_dir']
        os.makedirs(save_dir, exist_ok=True)
        out_dir = os.path.join(save_dir, "pred_shards")
        writer = PredictWriter(out_dir)

        trainer = L.Trainer(**config['trainer'], callbacks=[writer])
        trainer.predict(model, datamodule=datamodule, return_predictions=False)
        collate_preds(
            data_dir=out_dir,
            batch_size=config['datamodule']['batch_size'],
            num_samples=config['dataset']['neg_num'] + 1
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the fine-tuned model for the hyperedge prediction task.')

    parser.add_argument(
        '--config',
        type=Path,
        default='digitalcell/tasks/hyperedge/test_config.yaml',
        help='Path to the hyperedge prediction test configuration file.'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    
    main(config)