import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data_norm import DataNorm, DataNormSubMeanMinMax
from global_vars import dir_generated_files
from utils import get_strat_df, initial_columns

with open(os.path.join(dir_generated_files, "list_cols_predict"), "rb") as f:
    list_cols_predict = pickle.load(f)  # noqa: S301

x_cols = initial_columns + list_cols_predict
y_cols = ["Target"]


class TimeSeriesDataset(torch.utils.data.Dataset):  # type: ignore  # noqa: PGH003
    def __init__(self, x: torch.Tensor, y_true: torch.Tensor, num_data_points: int) -> None:
        self.x = x
        self.y_true = y_true
        self.num_data_points = num_data_points

    def __len__(self) -> int:
        return len(self.y_true) - self.num_data_points - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx : idx + self.num_data_points]
        y = self.y_true[idx + self.num_data_points]

        return x, y


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, version: str, **kwargs: Any) -> None:
        super().__init__()
        self.batch_size = kwargs.get("batch_size", 32)

        if version == "v1":
            self.data_processing = DataProcessingV1(**kwargs)
        else:
            msg = "Version not found."
            raise ValueError(msg)

        self.train_dataset, self.test_dataset = self.data_processing.get_datasets()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class DataProcessing(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.agg_time_min = kwargs.get("agg_time_min", 1440)
        self.predict_minutes_ahead = int(7 * 24 * 60 / self.agg_time_min)
        self.num_data_points = kwargs.get("num_data_points", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.train_pct = kwargs.get("train_pct", 0.8)
        self.x_cols = kwargs.get("x_cols", initial_columns + list_cols_predict)
        self.y_cols = kwargs.get("y_cols", ["Target"])

    @abstractmethod
    def get_datasets(self) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
        pass

    @property
    @abstractmethod
    def normalize_obj(self) -> DataNorm:
        pass

    @abstractmethod
    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        pass


class DataProcessingV1(DataProcessing):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dict_metrics = {}
        self.normalize_obj_instance = DataNormSubMeanMinMax(**kwargs)

    @property
    def normalize_obj(self) -> DataNorm:
        return self.normalize_obj_instance

    def get_datasets(self) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
        df_strats = get_strat_df(self.agg_time_min)
        df_strats = df_strats[initial_columns + list_cols_predict]
        dict_metrics = {}

        # Apply diff to initial columns
        for col in df_strats.columns:
            df_strats[f"{col}"] = df_strats[col].diff()

        df_strats["Target"] = df_strats["Close"].shift(-self.predict_minutes_ahead)

        df_strats = df_strats.dropna()

        for col in df_strats.columns:
            processed_col, kwargs = self.normalize_obj.normalize(df_strats[col])
            dict_metrics[col] = kwargs
            df_strats[col] = processed_col

        self.dict_metrics = dict_metrics

        df_train = df_strats.iloc[: int(len(df_strats) * self.train_pct)]
        df_test = df_strats.iloc[int(len(df_strats) * self.train_pct) :]

        train_dataset = TimeSeriesDataset(torch.from_numpy(df_train[x_cols].values).float(), torch.from_numpy(df_train[y_cols].values).float(), self.num_data_points)

        test_dataset = TimeSeriesDataset(torch.from_numpy(df_test[x_cols].values).float(), torch.from_numpy(df_test[y_cols].values).float(), self.num_data_points)

        return train_dataset, test_dataset

    def denormalize_target(self, y: torch.Tensor) -> torch.Tensor:
        return self.normalize_obj.denormalize(y, **self.dict_metrics["Target"])
