import argparse
import os
from pathlib import Path
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from digitalcell.tasks.enhancement.model import Enhance_Model
from digitalcell.tasks.enhancement.datamodule import Enhance_DataModule
from digitalcell.config import HiCT_Config
from digitalcell.utils import load_config

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

    print('Instantiating Enhance_Model... :)')
    model = Enhance_Model.load_from_checkpoint(
        config['ckpt_path'],
        config=HiCT_Config(**config['model']),
        map_location='cpu'
    )
    
    checkpoint_callback = ModelCheckpoint(
        save_last=True,              # <-- always save last epoch
        save_top_k=-1,                # <-- disable best-k checkpoints
        every_n_epochs=1000,            # save every epoch (overwrites)
        save_weights_only=False,
    )

    print('Instantiating EnhanceDataModule... :)')
    datamodule = Enhance_DataModule(**config['datamodule'], dataset_config=config['dataset'])

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

    parser = argparse.ArgumentParser(description='Train a model for the enhancement.')

    parser.add_argument(
        '--config',
        type=Path,
        default='digitalcell/tasks/enhancement/res_enhancement.yaml',
        help='Path to the enhancement configuration file.'
    )

    args = parser.parse_args()
    config = load_config(args.config)

    main(config)
    