import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from dataclasses import dataclass, field
from digitalcell.model.hict import HiCT
from digitalcell.optim.lr_scheduler import configure_scheduler


@dataclass
class HyperedgeModelConfig:

    backbone_path: str
   
    # Encoder parameters
    task_head_model: str = 'mlp'
    hidden_size: int = 1024
    activation: str = 'relu'
    num_layers: int = 4

    # Optimization parameters
    optim: dict = field(default_factory=dict)

class HyperedgeModel(L.LightningModule):

    def __init__(
        self,
        config,
    ):

        super().__init__()

        self.config = config
        self.save_hyperparameters(ignore=['backbone_model'])

        self.backbone = HiCT.load_from_checkpoint(config.backbone_path, map_location="cpu")
        self.backbone_dim = self.backbone.config.d_model

        class FeedForwardNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(FeedForwardNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                skip = x 
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x += skip
                return x

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.backbone_dim,
                nhead=self.backbone.config.num_heads,
                dim_feedforward=self.backbone.config.dim_feedforward,
                dropout=self.backbone.config.dropout,
                activation=self.backbone.config.activation,
                batch_first=True
            ),
            num_layers=self.config.num_layers
        )
        self.output_layer = nn.Linear(self.backbone_dim, 1)

        self.layer_norm = nn.LayerNorm(self.backbone_dim, elementwise_affine=False)


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
        
        # Freeze backbone except for the trainable portions
        encoder_layers = torch.arange(self.backbone.config.num_layers)
        trainable_layers = encoder_layers[self.config.optim.get('trainable_layers', [])]
        for name, param in self.backbone.named_parameters():
            # Check if this parameter belongs to any of the trainable layers
            is_trainable = any(f'encoder.layers.{layer_idx}.' in name for layer_idx in trainable_layers)
            param.requires_grad = is_trainable

        is_input_trainable = self.config.optim.get('train_input_layer', False)
        self.backbone._linear_projection.weight.requires_grad = is_input_trainable
        self.backbone._linear_projection.bias.requires_grad = is_input_trainable
        self.backbone.train()

        optimizer = optim.Adam(self.parameters(), lr=float(self.config.optim['lr']), betas=(0.9, 0.98), eps=1e-9)

        scheduler = configure_scheduler(self.config, optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}