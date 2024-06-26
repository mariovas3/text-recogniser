import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from lightning.pytorch.loggers import Logger

# the noqa flag is to tell the linter to ignore unused import;
import model_package.data  # noqa: F401
from model_package.lit_models.lit_transformer import LitResNetTransformer

# <start> hacky way to get wandb to watch grads in a thread safe way;
# look at the callbacks.py/SetLoggerWatch callback for this.
# class MyTrainer(L.Trainer):
#     def __init__(self, *args, **kwargs):
#         self.start_logging = True
#         super().__init__(*args, **kwargs)
# <end> hacky way to get wandb to watch grads in a thread safe way;


class MyLitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # fix data.max_length and model.max_seq_length to be the same at parse time;
        # this is the same as output dim;
        parser.link_arguments(
            "data.output_dims",
            "model.max_seq_length",
            compute_fn=lambda x: x[0],
            apply_on="instantiate",
        )
        # data.input_dims is passed to model constructor
        # after data is intstantiated and data.input_dims is set.
        parser.link_arguments(
            "data.input_dims", "model.input_dims", apply_on="instantiate"
        )
        # link idx_to_char with the datamodule's idx_to_char on instantiation;
        parser.link_arguments(
            "data.idx_to_char", "model.idx_to_char", apply_on="instantiate"
        )
        # make ModelCheckpoint configurable;
        parser.add_lightning_class_args(ModelCheckpoint, "my_model_checkpoint")
        parser.set_defaults(
            {
                "my_model_checkpoint.monitor": "validation/loss",
                "my_model_checkpoint.mode": "min",
                "my_model_checkpoint.every_n_epochs": 50,
            }
        )


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})


def main():
    cli = MyLitCLI(
        model_class=LitResNetTransformer,
        # <start> hacky way to get wandb to watch grads in a thread safe way;
        # trainer_class=MyTrainer,
        # <end> hacky way to get wandb to watch grads in a thread safe way;
        save_config_callback=LoggerSaveConfigCallback,
        seed_everything_default=0,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    main()
