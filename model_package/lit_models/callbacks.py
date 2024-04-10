from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

import wandb


class LogTrainPredsCallback(Callback):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        predicate = (
            (pl_module.current_epoch + 1) % trainer.check_val_every_n_epoch
            == 0
            and trainer.overfit_batches > 0
            and batch_idx == 0
        )
        if predicate:
            # log 10 images alongside ground truth and predicted labels;
            wandb_logger = trainer.logger
            idx_to_char = trainer.datamodule.idx_to_char
            n = 10
            assert len(pl_module.training_step_outputs) == 1

            outputs = pl_module.training_step_outputs[0]
            x, y = batch
            assert len(outputs) == len(y)
            # ignore BLANK, START, END AND PAD tokens;
            ground_truth_text, pred_text = [], []
            for yi, yi_pred in zip(y[:n], outputs[:n]):
                gtt_i = "".join(
                    idx_to_char[yii.item()] for yii in yi if yii > 3
                )
                pt_i = "".join(
                    idx_to_char[yii.item()] for yii in yi_pred if yii > 3
                )
                ground_truth_text.append(gtt_i)
                pred_text.append(pt_i)

            # log predictions as a table;
            columns = ["image", "ground_truth", "prediction", "epoch"]
            data = [
                [
                    wandb.Image(x_i),
                    y_i,
                    y_pred,
                    str(pl_module.current_epoch + 1),
                ]
                for x_i, y_i, y_pred in list(
                    zip(x[:n], ground_truth_text, pred_text)
                )
            ]
            wandb_logger.log_table(
                key="overfit_batch_table", columns=columns, data=data
            )


class LogPredsCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        lit_module: LightningModule,
        preds,
        batch,
        batch_idx,
    ):
        # preds is returned from validation_step;

        # let's log 10 sample img predictions from first batch;
        if trainer.overfit_batches == 0 and batch_idx == 0:
            wandb_logger = trainer.logger
            idx_to_char = trainer.datamodule.idx_to_char
            n = 10
            x, y = batch
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
            columns = ["image", "ground truth", "prediction", "epoch"]
            data = [
                [
                    wandb.Image(x_i),
                    y_i,
                    y_pred,
                    str(lit_module.current_epoch + 1),
                ]
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
