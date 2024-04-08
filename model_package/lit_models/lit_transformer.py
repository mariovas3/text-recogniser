from itertools import chain
from typing import Dict, Sequence, Union

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from model_package.models.metrics import MyCharErrorRate
from model_package.models.resnet import ResNet, get_final_height_width
from model_package.models.transformer_utils import *

# this config overfits a single 64-sized batch
# EMNISTLines in 200 train steps with Adam lr=1e-3;
TF_DIM = 64
TF_FC_DIM = 128
TF_DROPOUT = 0
TF_NUM_LAYERS = 4
TF_NHEAD = 4
LR = 1e-3


class LitResNetTransformer(L.LightningModule):
    def __init__(
        self,
        idx_to_char: Sequence[str],
        input_dims: Sequence[int],
        max_seq_length: int,
        resnet_config: Dict[str, Sequence[int]],
        tf_dim: int = TF_DIM,
        tf_fc_dim: int = TF_FC_DIM,
        tf_nhead: int = TF_NHEAD,
        tf_dropout: Union[int, float] = TF_DROPOUT,
        tf_num_layers: int = TF_NUM_LAYERS,
        lr: Union[int, float] = LR,
        with_enc_pos: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = len(idx_to_char)
        self.input_dims = input_dims
        self.idx_to_char = idx_to_char
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.start_token = self.char_to_idx["<START>"]
        self.end_token = self.char_to_idx["<END>"]
        self.padding_token = self.char_to_idx["<PAD>"]
        self.max_seq_length = max_seq_length

        self.d_model = tf_dim
        # the output channels of the resnet should equal the model dim;
        assert resnet_config["out_channels"][-1] == self.d_model
        self.lr = lr
        self.with_enc_pos = with_enc_pos

        self.test_cer = MyCharErrorRate(
            [self.padding_token, self.start_token, self.end_token]
        )
        self.val_cer = MyCharErrorRate(
            [self.padding_token, self.start_token, self.end_token]
        )

        # Encoder setup - RESNET;
        self.encoder = ResNet(resnet_config)
        # position embeds for encoder didn't seem to matter;
        if self.with_enc_pos:
            h_out, w_out = get_final_height_width(
                self.input_dims[-2], self.input_dims[-1], resnet_config
            )
            self.enc_pos_emb = PosEmbed(h_out * w_out, self.d_model)

        # Decoder setup - TransformerDecoder;
        self.embedding = nn.Embedding(self.num_classes, self.d_model)
        self.dec_pos_emb = PosEmbed(self.max_seq_length, self.d_model)
        self.y_mask = get_torch_mask(self.max_seq_length)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=tf_nhead,
                dim_feedforward=tf_fc_dim,
                dropout=tf_dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=tf_num_layers,
            # apply norm on final output of decoder;
            norm=nn.LayerNorm(self.d_model),
        )

        # classification head;
        # if the token embedding was used instead, we intuitively
        # want the repr for the current token to be as close as
        # possible to the correct next token - corresponding to
        # wanting to learn an exact identity mapping;
        # with a separate affine layer, we don't have
        # the same constraint.
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def configure_optimizers(self):
        if self.with_enc_pos:
            return torch.optim.Adam(
                chain(
                    self.encoder.parameters(),
                    self.enc_pos_emb.parameters(),
                    self.decoder.parameters(),
                    self.embedding.parameters(),
                    self.dec_pos_emb.parameters(),
                    self.classifier.parameters(),
                ),
                lr=self.lr,
            )
        return torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.embedding.parameters(),
                self.dec_pos_emb.parameters(),
                self.classifier.parameters(),
            ),
            lr=self.lr,
        )

    def forward(self, x):
        B = len(x)
        x = self.encode(x)

        output_tokens = (
            torch.ones((B, self.max_seq_length)) * self.padding_token
        ).long()
        output_tokens[:, 0] = self.start_token
        for Sq in range(1, self.max_seq_length):
            tgt = output_tokens[:, :Sq]
            # output is (B, num_classes, Sq)
            output = self.decode(x, tgt).argmax(dim=-2)  # -> (B, Sq)
            output_tokens[:, Sq] = output[:, -1]

            # stop if predicted only end tokens and padding tokens;
            if (
                (output_tokens[:, Sq] == self.end_token)
                | (output_tokens[:, Sq] == self.padding_token)
            ).all():
                break

        # make sure everything after end token or padding token is padding token;
        # this is because in the decoding above, we stop decoding only when
        # ALL elements of the batch output end tokens or padding tokens;
        # in the beginning when model not trained we can decode tokens not in (end, pad)
        # after end or pad; Also start from idx 2 since 0 is start token;
        for Sq in range(2, self.max_seq_length):
            ind = (output_tokens[:, Sq - 1] == self.end_token) | (
                output_tokens[:, Sq - 1] == self.padding_token
            )
            output_tokens[ind, Sq] = self.padding_token
        return output_tokens  # (B, Sq)

    def _get_outs_loss(self, batch):
        x, y = batch
        x = self.encode(x)
        # try predicting next token from current ones;
        outs = self.decode(x, y[:, :-1])
        # tried ignore_index=self.padding_token but got worse performance;
        loss = F.cross_entropy(outs, y[:, 1:], reduction="mean")
        return outs, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._get_outs_loss(batch)
        self.log(
            "training/loss",
            loss.item(),
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outs, loss = self._get_outs_loss(batch)
        # get preds from logits;
        outs = outs.argmax(-2)
        cer = self.val_cer(outs, batch[-1])
        m = {"validation/loss": loss.item(), "validation/cer": cer.item()}
        self.log_dict(m, logger=True, on_epoch=True, prog_bar=True)
        return outs

    def test_step(self, batch, batch_idx):
        outs, loss = self._get_outs_loss(batch)
        cer = self.test_cer(outs.argmax(-2), batch[-1])
        m = {"test/loss": loss.item(), "test/cer": cer.item()}
        self.log_dict(m, logger=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return self(batch[0])

    def encode(self, x):
        """
        Encodes image using ResNet encoder.

        Args:
            x: shape (B, C_in, H_in, W_in).

        Returns:
            Tensor of shape (B, S_mem, C_out) where S_mem = H_out * W_out
            and C_out is equal to self.d_model.
        """
        out = torch.flatten(self.encoder(x), 2).transpose(-2, -1)
        if self.with_enc_pos:
            return self.enc_pos_emb(out)
        return out

    def decode(self, memory, tgt):
        """
        Does Transformer Decoder pass.

        Args:
            memory: tensor of shape (B, Sk, d_model), where
                Sk = H_out * W_out and d_model = C_out holds.
                These are the encoded images from the resnet.
            tgt: tensor of shape (B, Sq) the query tokens.

        Returns:
            Tensor of shape (B, C, Sq), where C is the num classes.
        """
        # True values of mask are ignored;
        tgt_padding_mask = tgt == self.padding_token

        # get tgt embeddings;
        tgt = self.dec_pos_emb(self.embedding(tgt))
        Sq = tgt.size(-2)
        tgt_mask = self.y_mask[:Sq, :Sq]
        # out is (B, Sq, d_model)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        # (B, Sq, d_model) -> (B, Sq, C) -> (B, C, Sq)
        return self.classifier(out).transpose(-2, -1)

    @staticmethod
    def add_to_argparse(parser):
        # here only for testing purposes;
        parser.add_argument("--tf_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_fc_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_dropout", type=float, default=TF_DROPOUT)
        parser.add_argument("--tf_num_layers", type=int, default=TF_NUM_LAYERS)
        parser.add_argument("--tf_nhead", type=int, default=TF_NHEAD)
        parser.add_argument("--lr", type=float, default=LR)
        # model seemed to work without encoder pos embeds.
        parser.add_argument(
            "--with_enc_pos", action="store_true", default=False
        )
        return parser
