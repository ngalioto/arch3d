import torch
import torch.optim as optim

from digitalcell.model.hict import HiCT, HiCT_Config
from digitalcell.optim.lr_scheduler import configure_scheduler
from digitalcell.data.dataset import HiC_Sequence

class Enhance_Model(HiCT):
    """
    Just a wrapper around HiCT for the enhance task.
    """
    def __init__(
        self, 
        config: HiCT_Config,
    ) -> None:
        
        super().__init__(config=config)
        self.save_hyperparameters()
    
    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probs = self.pretext_task_head(embeddings)
        
        N = embeddings.shape[-2]
        rows, cols = torch.triu_indices(N, N, offset=0) # include main diagonal
        
        # masked_loci.values are (B, N, N)
        targets = targets[:, rows, cols]

        return self._loss(probs, targets).squeeze()
    
    def loss(
        self,
        hic_seq: HiC_Sequence,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        
        embeddings = self.forward(hic_seq)

        # Compute the loss
        loss = self._compute_loss(embeddings, targets)

        return loss
    
    def training_step(
        self,
        batch: tuple[HiC_Sequence, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(*batch)

        # Log the learning rate and training loss
        current_lr = self.lr_schedulers().get_last_lr()[0]
        self.log("Learning rate", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True)
        self.log("Training loss", loss.detach(), on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)

        return loss
    
    def validation_step(
        self,
        batch: tuple[HiC_Sequence, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(*batch)

        # Log the learning rate and training loss
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch), sync_dist=True)


    def configure_optimizers(
        self,
    ) -> dict[str, optim.Optimizer | optim.lr_scheduler.LRScheduler]:
        
        # Freeze backbone except for the trainable portions
        encoder_layers = torch.arange(self.config.num_layers)
        trainable_layers = encoder_layers[self.config.optim.get('trainable_layers', [])]
        trainable_params = []
        for name, param in self.named_parameters():
            # Check if this parameter belongs to any of the trainable layers
            is_trainable = any(f'encoder.layers.{layer_idx}.' in name for layer_idx in trainable_layers)
            param.requires_grad = is_trainable
            if is_trainable:
                trainable_params.append(param)

        is_input_trainable = self.config.optim.get('train_input_layer', False)
        for param in self._linear_projection.parameters():
            param.requires_grad = is_input_trainable
            if is_input_trainable:
                trainable_params.append(param)
        is_pretext_task_head_trainable = self.config.optim.get('train_pretext_task_head', False)
        for param in self.pretext_task_head.parameters():
            param.requires_grad = is_pretext_task_head_trainable
            if is_pretext_task_head_trainable:
                trainable_params.append(param)

        optimizer = optim.Adam(trainable_params, lr=float(self.config.optim['lr']), betas=(0.9, 0.98), eps=1e-9)

        scheduler = configure_scheduler(self.config, optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}