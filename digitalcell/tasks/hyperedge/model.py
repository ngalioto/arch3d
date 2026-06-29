import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from dataclasses import dataclass, field
from digitalcell.model.hict import HiCT
from digitalcell.optim.lr_scheduler import configure_scheduler


@dataclass
class HyperedgeModelConfig:

    # Path to the pretrained HiCT checkpoint. The backbone is NOT run at train
    # time -- locus embeddings are precomputed offline by
    # scripts/generate_embeddings.py. The checkpoint is read only to recover the
    # encoder dimensions (d_model, num_heads, dim_feedforward, dropout,
    # activation) used to build the task head.
    backbone_path: str

    # Task-head parameters. The task head is a stack of `num_layers` transformer
    # encoder layers applied to the precomputed locus embeddings.
    num_layers: int = 4

    # d_model used ONLY by the 'inverse_sqrt' (Noam) learning-rate scheduler.
    # Ignored by every other scheduler and does NOT size the task head.
    hidden_size: int = 1024

    # Optimization parameters
    optim: dict = field(default_factory=dict)

class HyperedgeModel(L.LightningModule):

    def __init__(
        self,
        config,
    ):

        super().__init__()

        self.config = config
        self.save_hyperparameters()

        # The backbone is never run at train time -- locus embeddings are
        # precomputed offline (see scripts/generate_embeddings.py). Load the
        # checkpoint only to recover the encoder architecture, then drop it so it
        # is NOT registered as a submodule. Keeping it would replicate the frozen
        # backbone on every GPU, bloat every checkpoint, and force
        # find_unused_parameters under DDP.
        # weights_only=False: torch>=2.6 defaults to True, which rejects the pickled
        # HiCT_Config in our own (trusted) checkpoint.
        backbone = HiCT.load_from_checkpoint(config.backbone_path, map_location="cpu", weights_only=False)
        backbone_config = backbone.config
        self.backbone_dim = backbone_config.d_model
        del backbone

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.backbone_dim,
                nhead=backbone_config.num_heads,
                dim_feedforward=backbone_config.dim_feedforward,
                dropout=backbone_config.dropout,
                activation=backbone_config.activation,
                batch_first=True
            ),
            num_layers=self.config.num_layers
        )
        self.output_layer = nn.Linear(self.backbone_dim, 1)


    def forward(
        self, 
        locus_embeddings: torch.Tensor,
        size_list: torch.Tensor
    ) -> torch.Tensor:
        # Forward pass through the model

        batch_size = locus_embeddings.shape[0]
        num_edges = size_list.shape[-1]
        
        logits = torch.zeros(batch_size, num_edges, device=self.device)
        for batch_idx in range(batch_size):
            for edge_idx in range(num_edges):
                edge_size = size_list[batch_idx, edge_idx]
                edge_nodes = locus_embeddings[batch_idx, edge_idx, :edge_size].unsqueeze(0)

                edge_nodes = self.transformer_encoder(edge_nodes)

                logits[batch_idx, edge_idx] = torch.mean(self.output_layer(edge_nodes).squeeze(-1))
               
     

        return logits

    def _loss(
        self, 
        yhat: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:

        return torch.nn.functional.binary_cross_entropy_with_logits(yhat, y, weight=weights)

    def training_step(self, batch, batch_idx):

        x, y, weights, size_list, _, _ = batch
        
        batch_size = size_list.shape[0]

        yhat = self(x, size_list) # logits

        loss = self._loss(yhat, y, weights)

        current_lr = self.lr_schedulers().get_last_lr()[0]
        self.log("Learning rate", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, weights, size_list, _, _ = batch
        
        batch_size = size_list.shape[0]
        
        yhat = self(x, size_list) # logits
        loss = self._loss(yhat, y, weights)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)


    def predict_step(self, batch, batch_idx):
        embeddings, labels, weights, size_list, hyperedges, data_names = batch
        
        # Forward pass
        predictions = self(embeddings, size_list)
        probs = torch.sigmoid(predictions)
        
        return {
            'probs': probs,
            'labels': labels,
            'weights': weights,
            'hyperedges': hyperedges,
            'size_list': size_list,
            'data_names': data_names,
            'batch_idx': batch_idx
        }

    def configure_optimizers(
        self
    ) -> dict[str, optim.Optimizer | optim.lr_scheduler.LRScheduler]:

        # Only the task head (transformer encoder + output layer) is trainable.
        # The backbone is not part of this module (embeddings are precomputed),
        # so every parameter here participates in the forward/backward pass.
        optimizer = optim.Adam(self.parameters(), lr=float(self.config.optim['lr']), betas=(0.9, 0.98), eps=1e-9)

        scheduler = configure_scheduler(self.config, optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}