from dataclasses import dataclass

import torch
from torch import nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.layers.attention import FullAttention, AttentionLayer
from prism.layers.embed import DataEmbedding_inverted
from prism.layers.transformer import Encoder, EncoderLayer
from prism.models.base import Model, models


@dataclass(frozen=True, slots=True)
class iTransformerConfig(ModelConfig):
    activation: str = "gelu"
    d_ff: int = 2048
    d_model: int = 512
    dropout: float = 0.1
    e_layers: int = 2
    embed: str = "timeF"
    factor: int = 1
    n_heads: int = 8


@models.register("iTransformer")
class iTransformer(Model):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, config: iTransformerConfig, dims: Dims) -> None:
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(config.seq_len, config.d_model, config.embed, config.freq,
                                                    config.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=False), config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        # Decoder
        self.projection = nn.Linear(config.d_model, config.pred_len, bias=True)

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return iTransformerConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def forward(self, batch: Batch) -> Forecast:
        x_enc = batch.x
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return Point(dec_out[:, -self.pred_len:, :])  # [B, L, D]
