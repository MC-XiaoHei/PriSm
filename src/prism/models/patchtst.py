from dataclasses import dataclass

import torch
from torch import nn

from prism.config import Dims, ModelConfig
from prism.fields import Batch, Forecast, Point
from prism.layers.attention import FullAttention, AttentionLayer
from prism.layers.embed import PatchEmbedding
from prism.layers.transformer import Encoder, EncoderLayer
from prism.models.base import Model, models


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


@dataclass(frozen=True, slots=True)
class PatchTSTConfig(ModelConfig):
    activation: str = "gelu"
    d_ff: int = 2048
    d_model: int = 512
    dropout: float = 0.1
    e_layers: int = 2
    factor: int = 1
    n_heads: int = 8


@models.register("PatchTST")
class PatchTST(Model):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, config: PatchTSTConfig, dims: Dims, patch_len: int = 16, stride: int = 8) -> None:
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            config.d_model, patch_len, stride, padding, config.dropout)

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
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = config.d_model * \
                       int((config.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(dims.enc_in, self.head_nf, config.pred_len,
                                head_dropout=config.dropout)

    @classmethod
    def config_type(cls) -> type[ModelConfig]:
        return PatchTSTConfig

    @classmethod
    def build(cls, config: ModelConfig, dims: Dims) -> Model:
        return cls(config, dims)

    def forward(self, batch: Batch) -> Forecast:
        x_enc = batch.x
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return Point(dec_out[:, -self.pred_len:, :])  # [B, L, D]
