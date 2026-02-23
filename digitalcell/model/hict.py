import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from dataclasses import dataclass, field
from digitalcell.data import constants
from digitalcell.data.dataset import HiC_Sequence, Locus_Position
from digitalcell.optim.lr_scheduler import configure_scheduler
from typing import Callable

"""
Hi-C Transformer (HiCT)
"""

@dataclass
class HiCT_Config:
   
    # Encoder parameters
    d_model: int = 768
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = 'relu'
    num_layers: int = 12
    prepend_cls: bool = False

    # Embedding parameters
    embedding_hidden_size: int = 1024
    use_gaussian_fourier_features: bool = False
    fourier_feature_scale: float = 1000 # must be positive

    # Optimization parameters
    optim: dict = field(default_factory=dict)

class HiCT(L.LightningModule):

    def __init__(
        self,
        config: HiCT_Config ,
    ):
        
        super().__init__()
        self.config = config

        config.activation = self._check_activation_arg(config.activation)

        ########################
        # Build the vocabulary #
        ########################
        self.special_tokens = {'class': '[CLS]', 'mask': '[MASK]', 'pad': '[PAD]'}
        self.special_token_ids = {}
        self.special_token_embedding_vectors = nn.Embedding(len(self.special_tokens), config.d_model)

        #########################
        # Define the embeddings #
        #########################
        self.chromosome_embedding_vectors = nn.Embedding(constants.NUM_CHROM, config.d_model)

        # Instantiate Gaussian matrix for Fourier features here
        if config.use_gaussian_fourier_features:
            # Gaussian embedding matrix
            self.gaussian_embedding = self.config.fourier_feature_scale * torch.randn(1, config.d_model // 2)
        
        #################################
        # Build the transformer encoder #
        #################################
        self._linear_projection = nn.Linear(575010, config.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=config.num_layers,
        )

        ###############################
        # Build the pretext task head #
        ###############################
        self.pretext_task_head = self._build_pretext_task_head()
        self._loss = torch.nn.MSELoss()

        self.save_hyperparameters()

    def on_fit_start(self) -> None:
        """
        Initializes the special token IDs
        Lightning hook called upon fit start, allowing IDs to be initialized on the correct device
        """
        for token in self.special_tokens.values():
            self.special_token_ids[token] = torch.tensor(len(self.special_token_ids), dtype=torch.int32, device=self.device)

    def _check_activation_arg(
        self,
        activation: str
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        
        if type(activation) == str:
            if activation == 'relu':
                activation = nn.ReLU()
            elif activation == 'gelu':
                activation = nn.GELU()
            elif activation == 'leaky_relu':
                activation = nn.LeakyReLU()
            elif activation == 'tanh':
                activation = nn.Tanh()
            else:
                raise ValueError('Invalid activation function. Only "relu", "gelu", "leaky_relu", and "tanh" are supported.')
        return activation
    
    def _quadrature(
        self,
        points: torch.Tensor,
        weights: torch.Tensor | None
    ) -> torch.Tensor:
        
        """
        points: torch.Tensor
            Tensor of shape (*, num_samples, d_model)
        weights: torch.Tensor
            Tensor of shape (*, num_samples). If None, then the weights are uniform.

        Returns:
            A tensor of shape (*, d_model) containing the weighted sum of the inputs

        Here B is the batch, num_samples is the number of quadrature points, and d_model is the model dimension
        """

        if weights is None: # Compute the sample average
            return torch.mean(points, dim=-2, keepdim=False)
        else: # Compute the weighted sum
            return torch.einsum(
                '...jd, ...j -> ...d', 
                points,
                weights
            )
    
        
    def _fourier_features(
        self,
        d_model: int,
        position: torch.Tensor
    ) -> torch.Tensor:
        
        """
        d_model:
            The dimension of the model
        position:
            1D tensor of shape (*, N) where N is the length of the sequence

        Returns:
            A tensor of shape (*, N, d_model) containing the positional encoding

        """
        
        positional_embedding = torch.zeros(*position.shape, d_model, dtype=self.dtype, device=self.device)

        # gaussian embedding should be 1 x d_model
        if self.config.use_gaussian_fourier_features:
            val = 2 * torch.pi * (position.unsqueeze(-1) @ self.gaussian_embedding)
        else:
            val = 2 * torch.pi * position.unsqueeze(-1) / (self.config.fourier_feature_scale ** (torch.arange(0, d_model, 2, dtype=self.dtype, device=self.device) / d_model))
        positional_embedding[..., 0::2] = torch.sin(val)
        positional_embedding[..., 1::2] = torch.cos(val)

        return positional_embedding

    def _sine_encoding(
        self,
        d_model: int,
        position: torch.Tensor
    ) -> torch.Tensor:
        
        """
        d_model:
            The dimension of the model
        position:
            1D tensor of shape (*, N) where N is the length of the sequence

        Returns:
            A tensor of shape (*, N, d_model) containing the positional encoding
        """
        
        positional_embedding = torch.zeros(*position.shape, d_model, dtype=self.dtype, device=self.device)

        freq = 1000

        val = position.unsqueeze(-1) / (freq ** (torch.arange(0, d_model, 2, dtype=self.dtype, device=self.device) / d_model))
        positional_embedding[..., 0::2] = torch.sin(val)
        positional_embedding[..., 1::2] = torch.cos(val)

        return positional_embedding

    def _sine_positional_encoding(
        self,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

        """
        Computes the positional encoding of the starting and ending point of every locus and returns the concatenation

        d_model:
            The model dimension. int
        start_pos:
            The starting position of the locus. Tensor of shape (*, N)
        end_pos:
            The ending position of the locus. Tensor of shape (*, N)

        Returns:
            A tensor of shape (*, N, d_model) containing the positional encoding

        Here, N is the sequence length, and d_model is the model dimension        
        """        
        
        start_embedding = self._sine_encoding(self.config.d_model // 2, start_pos)
        end_embedding = self._sine_encoding(self.config.d_model // 2, end_pos)

        return torch.cat((start_embedding, end_embedding), dim=-1)
    
    def linear_projection(
        self,
        loci: Locus_Position
    ) -> torch.Tensor:
      
        return self._linear_projection(loci.values)

    def position_embedding(
        self,
        loci: Locus_Position,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        
        """
        Positional encoding for the transformer. Computes the sum of the base-pair encoding and the chromosome embedding.

        loci:
            A Locus_Position object. Contains the information needed to compute the positional encoding for a collection of loci.
        mask (optional):
            A boolean mask of shape (*, N) where N is the length of the sequence. If provided, only the loci where mask is `True` will be embedded.
        """

        if mask is None:
            return self._sine_positional_encoding(loci.start, loci.end) + self.chromosome_embedding_vectors(loci.chromosomes)
        else:
            return self._sine_positional_encoding(loci.start[mask], loci.end[mask]) + self.chromosome_embedding_vectors(loci.chromosomes[mask])

    def _embed_sequence(
        self,
        hic_seq: HiC_Sequence
    ) -> torch.Tensor:
        
        """
        Embeds all tokens (loci) into the model dimension

        hic_seq:
            A HiC_Sequence object
        
        Returns:
            A tensor of shape (B, N, d_model) containing the embeddings of the input loci
        """

        embedding = self.linear_projection(hic_seq.input_loci).type(self.dtype)

        if hic_seq.masked_loci is not None:
            # mask will flatten the tensors so need to reshape position_embedding to match
            embedding[hic_seq.input_loci.mask] = embedding[hic_seq.input_loci.mask] + self.position_embedding(hic_seq.input_loci).view(-1, self.config.d_model)

            # Embed the masked loci using the [MASK] special token
            mask_embedding = self.special_token_embedding_vectors(self.special_token_ids[self.special_tokens['mask']]).expand(*hic_seq.masked_loci.shape, -1)
            mask_embedding = mask_embedding + self.position_embedding(hic_seq.masked_loci)

            embedding[~hic_seq.input_loci.mask] = mask_embedding.view(-1, self.config.d_model)
        else:
            # mask is all True here, so no need for reshaping
            embedding = embedding + self.position_embedding(hic_seq.input_loci)

        return embedding
    
    def embed_sequence(
        self,
        hic_seq: HiC_Sequence
    ) -> torch.Tensor:
        
        """
        Embeds all tokens with the option to prepend the [CLS] token
        """
        
        if self.config.prepend_cls:
            num_batch, seq_len = hic_seq.input_loci.shape[:2]
            embedding = torch.zeros(num_batch, seq_len + 1, self.config.d_model, dtype=self.dtype, device=self.device)
            embedding[:, 1:] = self._embed_sequence(hic_seq)
            embedding[:, 0] = self.special_token_embedding_vectors(self.special_token_ids[self.special_tokens['class']])
        else:
            embedding = self._embed_sequence(hic_seq)

        return embedding
    
    def _build_pretext_task_head(
        self,
    ) -> nn.Module:
        
        """
        Neural network for predicting loci contacts from the model output
        """
        
        class MaskPredictionHead(nn.Module):

            def __init__(
                self,
                config: dict
            ):
                
                super().__init__()
                self.d_model = config.d_model
                self.num_heads = config.num_heads

                self.output_layer = nn.Sequential(
                    nn.Linear(config.d_model, 2 * config.d_model),
                    nn.ReLU(),
                    nn.Linear(2*config.d_model, 2*config.d_model),
                    nn.ReLU(),
                    nn.Linear(2*config.d_model, config.d_model),
                    nn.ReLU(),
                    nn.Linear(config.d_model, 1)
                )
            

            def forward(
                self,
                masked_loci: torch.Tensor
            ) -> torch.Tensor:
                
                """
                masked_loci:
                    A tensor of shape (B, N, d) containing the model output corresponding to masked loci

                Returns:
                    A tensor of shape (B, N(N+1)/2) containing the output of the pretext task head

                B is the batch, N is the number of masked tokens, and d is the model dimension.
                """

                N = masked_loci.shape[-2]
    
                # Get indices for upper triangular portion (including diagonal)
                rows, cols = torch.triu_indices(N, N, offset=0)
                
                return self.output_layer(masked_loci[:, rows] + masked_loci[:, cols]).squeeze(-1)
            
        return MaskPredictionHead(self.config)
    
    def _compute_loss(
        self,
        hic_seq: HiC_Sequence,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        
        """
        hic_seq:
            A HiC_Sequence object
        encoding:
            A tensor of shape (B, N, d) containing the embeddings of the masked loci

        Returns:
            A scalar tensor containing the binary cross-entropy loss

        B is the batch, N is the number of masked tokens, and d is the model dimension.
        """

        probs = self.pretext_task_head(encoding)

        N = encoding.shape[-2]
        rows, cols = torch.triu_indices(N, N, offset=0) # include main diagonal
        
        # masked_loci.values are (B, N, N)
        targets = hic_seq.masked_loci.values[:, rows, cols]

        return self._loss(probs, targets).squeeze()
    
    def loss(
        self,
        hic_seq: HiC_Sequence
    ) -> torch.Tensor:
        
        """
        Parameters:
        hic_seq: HiC_Sequence
            The HiC_Sequence object over which the loss will be computed.

        Returns:
            A scalar tensor containing the loss value.
        """
        
        # Embed the sequence of tokens
        embeddings = self.embed_sequence(hic_seq)

        # Forward pass through the encoder
        encoding = self.encoder(embeddings)

        # Get only the loci that enter the loss 
        encoding = encoding[hic_seq.masked_loci.mask].view(hic_seq.input_loci.shape[0], -1, self.config.d_model)

        # Compute the loss
        loss = self._compute_loss(hic_seq, encoding)

        return loss

    
    # Lightning methods:

    def forward(
        self,
        hic_seq: HiC_Sequence
    ) -> torch.Tensor:
        
        """
        Evaluates the model without the decoder
        """

        embeddings = self.embed_sequence(hic_seq)
        return self.encoder(embeddings)
    
    def training_step(
        self,
        batch: HiC_Sequence,
        batch_idx: int
    ) -> torch.Tensor:

        loss = self.loss(batch)

        # Log the learning rate and training loss
        current_lr = self.lr_schedulers().get_last_lr()[0]
        self.log("Learning rate", current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("Training loss", loss.detach(), on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int
    ) -> None:

        """
        Computes validation metrics

        Lightning automatically disables gradients when calling this method
        """

        loss = self.loss(batch)

        self.log("Validation loss", loss.detach(), prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True)
    
    def configure_optimizers(
        self
    ) -> dict[str, optim.Optimizer | optim.lr_scheduler.LRScheduler]:

        optimizer = optim.Adam(self.parameters(), lr=float(self.config.optim['lr']), betas=(0.9, 0.98), eps=1e-9)

        scheduler = configure_scheduler(self.config, optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}