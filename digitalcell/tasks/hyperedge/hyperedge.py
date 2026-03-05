import argparse
import os
from pathlib import Path

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from digitalcell.conf.utils import load_config
from digitalcell.tasks.hyperedge.datamodule import HyperedgeDataModule
from digitalcell.tasks.hyperedge.model import HyperedgeModel, HyperedgeModelConfig

# include when using Tensor Cores
torch.set_float32_matmul_precision('high')


@rank_zero_only
def save_config_file(config: dict) -> None:

    save_dir = config['datamodule']['metadata_dir']
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, 'config.yaml')
        with open(fname, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

def main(
    config: dict
) -> None:
    
    if config['datamodule']['metadata_dir'] is not None:
        save_config_file(config)

    print('Instantiating HyperedgeModel... :)')
    model_config = HyperedgeModelConfig(**config['model'])
    model = HyperedgeModel(model_config)

    # Save checkpoint every epoch
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,              # Save the best model based on validation loss
        every_n_epochs=1,          # Save every epoch
        save_weights_only=False,    # Save full model (set True if you only want weights)
        save_last=True
    )

    print('Instantiating HyperedgeDataModule... :)')
    datamodule = HyperedgeDataModule(**config['datamodule'], dataset_config=config['dataset'])

    if config['logger']['use_wandb']:
        del config['logger']['use_wandb']
        logger = WandbLogger(**config['logger'])
    else:
        logger = None

    print('Instantiating Trainer... :)')
    trainer = L.Trainer(
        **config['trainer'],
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#fit
    trainer.fit(
        model=model,
        datamodule=datamodule
    )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model for the hyperedge prediction task.')

    parser.add_argument(
        '--config',
        type=Path,
        default='digitalcell/tasks/hyperedge/hyperedge_config.yaml',
        help='Path to the hyperedge configuration file.'
    )

    args = parser.parse_args()
    config = load_config(args.config)

    main(config)