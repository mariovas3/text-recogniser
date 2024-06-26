import torch
from torch import nn

from .resnet import ResNet, get_final_height_width
from .transformer_utils import *

# this config get 0 cer on a single 64-sized training batch
# in 200 train steps with Adam lr=1e-3;
TF_DIM = 64
TF_FC_DIM = 128
TF_DROPOUT = 0
TF_NUM_LAYERS = 4
TF_NHEAD = 4


class ResNetTransformer(nn.Module):
    """Same stuff as Attention is all you need; but encoder is a ResNet."""

    def __init__(self, data_config, resnet_config, args=None):
        super().__init__()
        self.num_classes = len(data_config["idx_to_char"])
        self.input_dims = data_config["input_dims"]
        self.idx_to_char = data_config["idx_to_char"]
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        self.start_token = self.char_to_idx["<START>"]
        self.end_token = self.char_to_idx["<END>"]
        self.padding_token = self.char_to_idx["<PAD>"]
        self.max_seq_length = data_config["max_seq_length"]

        self.args = {} if args is None else vars(args)
        self.d_model = self.args.get("tf_dim", TF_DIM)
        tf_fc_dim = self.args.get("tf_fc_dim", TF_FC_DIM)
        tf_nhead = self.args.get("tf_nhead", TF_NHEAD)
        tf_dropout = self.args.get("tf_dropout", TF_DROPOUT)
        tf_num_layers = self.args.get("tf_num_layers", TF_NUM_LAYERS)
        self.with_enc_pos = self.args.get("with_enc_pos", False)

        # Encoder setup - RESNET;
        self.encoder = ResNet(resnet_config)
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

    def forward(self, x):
        """
        Returns batch of output tokens idxs.

        Args:
            x (torch.Tensor): Batch of images (B x self.input_dims).
        Returns:
            out (torch.Tensor): Greedily decoded token idxs in shape
                (B, self.max_seq_length).
        """
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

    def encode(self, x):
        # self.encoder(x) gives shape (B, C_out, H_out, W_out)
        # flatten to (B, C_out, H_out * W_out) and permute so get
        # (B, H_out * W_out, C_out) - C_out is the model dim
        # and H_out * W_out is S_k - seq length of keys;
        out = torch.flatten(self.encoder(x), 2).transpose(-2, -1)
        if self.with_enc_pos:
            return self.enc_pos_emb(out)
        return out

    def decode(self, memory, tgt):
        # (B, S_q)
        # True values are ignored in attention;
        tgt_padding_mask = tgt == self.padding_token

        # get tgt embeddings
        tgt = self.dec_pos_emb(self.embedding(tgt))
        Sq = tgt.size(-2)
        tgt_mask = self.y_mask[:Sq, :Sq]
        # (B, Sq, d_model)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        # TODO remove the assert below after ready to deploy;
        assert not out.isnan().any()
        # (B, num_classes, Sq)
        # use separate affine layer to decode;
        return self.classifier(out).transpose(-2, -1)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tf_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_fc_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_dropout", type=float, default=TF_DROPOUT)
        parser.add_argument("--tf_num_layers", type=int, default=TF_NUM_LAYERS)
        parser.add_argument("--tf_nhead", type=int, default=TF_NHEAD)
        # model seemed to work without encoder pos embeds.
        parser.add_argument(
            "--with_enc_pos", action="store_true", default=False
        )
        return parser
