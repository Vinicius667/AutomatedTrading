from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim

from data_module import TimeSeriesDataModule


class LSTMModel(pl.LightningModule):
    def __init__(
        self,
        data_module: TimeSeriesDataModule,
        **kwargs: int,
    ) -> None:
        super().__init__()

        self.data_module = data_module
        self.input_size = len(data_module.data_processing.x_cols)
        self.output_size = len(data_module.data_processing.y_cols)
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_layers = kwargs.get("num_layers", 2)
        self.lr = kwargs.get("lr", 0.001)

        self.training_step_losses = []
        self.validation_step_losses = []
        self.validation_pct_diff = []
        self.acc = []
        self.min_validation_avg_pct_diff = 1e6
        self.max_validation_acc = 0

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Output size: batch_size x seq_length x hidden_size
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.loss = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        # Fully connected layer
        return self.fc(out)

    def training_step(self, batch: Tensor, batch_idx: int) -> dict[str, Tensor]:
        y_hat, y_true, loss = self._shared_foward_step(batch, batch_idx, "train")
        self.training_step_losses.append(loss)
        self.log("train_loss", loss, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch: Tensor, batch_idx: int) -> dict[str, Tensor]:
        y_hat, y_true, loss = self._shared_foward_step(batch, batch_idx, "val")
        pct_diff: Tensor = 100 * (((y_hat - y_true) / y_true).abs()).mean()
        self.validation_step_losses.append(loss)
        self.validation_pct_diff.append(pct_diff)

        # Check if sign of prediction is same as sign of true value
        target_metrics = self.data_module.data_processing.dict_metrics["Target"]
        reversed_y_hat = self.data_module.data_processing.normalize_obj_instance.denorm_func(y_hat, **target_metrics)
        reversed_y_true = self.data_module.data_processing.normalize_obj_instance.denorm_func(y_true, **target_metrics)

        right_market_direction = torch.sign(reversed_y_hat) == torch.sign(reversed_y_true)

        acc = right_market_direction.sum() / len(reversed_y_hat)

        self.acc.append(acc)

        self.log("val_loss", loss, prog_bar=False)
        return {"loss": loss}

    def _shared_foward_step(self, batch: Tensor, batch_idx: int, stage: str) -> tuple[Tensor, Tensor, Tensor]:
        x, y_true = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y_true)
        return y_hat, y_true, loss

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack(self.training_step_losses).mean()
        self.log("training_epoch_average_loss", avg_loss, prog_bar=True)
        self.training_step_losses.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack(self.validation_step_losses).mean()
        avg_pct_diff = torch.stack(self.validation_pct_diff).mean()
        avg_acc = torch.stack(self.acc).mean()
        self.min_validation_avg_pct_diff = min(
            self.min_validation_avg_pct_diff,
            avg_pct_diff.item(),
        )

        self.max_validation_acc = max(
            self.max_validation_acc,
            avg_acc.item(),
        )

        self.log("val_epoch_average_loss", avg_loss, prog_bar=True)
        self.log("val_epoch_average_pct_diff", avg_pct_diff, prog_bar=True)
        self.log("val_epoch_average_acc", avg_acc, prog_bar=True)
        self.validation_step_losses.clear()
        self.validation_pct_diff.clear()

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
