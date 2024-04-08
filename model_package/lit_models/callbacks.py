import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

# from model_package.metadata.emnist import MAPPING


class LogPredsCallback(Callback):
    def on_validation_batch_end(
        self, trainer: Trainer, lit_module, preds, batch, batch_idx
    ):
        wandb_logger = trainer.logger
        idx_to_char = trainer.datamodule.idx_to_char
        # preds comes from validation_step;

        # let's log 20 sample img predictions from first batch;
        if batch_idx == 0:
            n = 10
            x, y = batch
            imgs = [img for img in x[:n]]
            # ignore BLANK, START, END and PAD tokens;
            ground_truth_text, pred_text = [], []
            for yi, yi_pred in zip(y[:n], preds[:n]):
                gtt_i = "".join(
                    idx_to_char[yii.item()] for yii in yi if yii > 3
                )
                pt_i = "".join(
                    idx_to_char[yii.item()] for yii in yi_pred if yii > 3
                )
                ground_truth_text.append(gtt_i)
                pred_text.append(pt_i)

            # log predictions as a Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred]
                for x_i, y_i, y_pred in list(
                    zip(x[:n], ground_truth_text, pred_text)
                )
            ]
            wandb_logger.log_table(
                key="sample_table", columns=columns, data=data
            )


class SetLoggerWatch(Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        if hasattr(trainer, "start_logging"):
            if trainer.start_logging:
                trainer.logger.watch(pl_module)
                trainer.start_logging = False
