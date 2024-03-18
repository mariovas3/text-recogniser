import lightning as L
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger

from model_package.data.emnist_lines import EMNISTLines
from model_package.lit_models.lit_transformer import LitResNetTransformer


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})


def main():
    cli = LightningCLI(
        model_class=LitResNetTransformer,
        datamodule_class=EMNISTLines,
        save_config_callback=LoggerSaveConfigCallback,
        seed_everything_default=0,
    )


if __name__ == "__main__":
    main()
