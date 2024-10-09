from abc import ABC, abstractmethod
from typing import Any, Tuple

import pandas as pd
import torch

from data_norm_funcs import dict_norm_denorm_func_pairs


class DataNorm(ABC):
    @abstractmethod
    def normalize(self, x: pd.Series) -> tuple[torch.Tensor, dict]:
        pass

    @abstractmethod
    def denormalize(self, y: torch.Tensor, **denorm_kwargs: Any) -> torch.Tensor:
        pass


class DataNormSubMeanMinMax(DataNorm):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.norm_func, self.denorm_func = dict_norm_denorm_func_pairs["sub_mean_min_max"]

    def normalize(self, x: pd.Series) -> tuple[torch.Tensor, dict]:
        x_norm, denorm_kwargs = self.norm_func(x)
        return x_norm, denorm_kwargs

    def denormalize(self, y: torch.Tensor, **denorm_kwargs: Any) -> torch.Tensor:
        return self.denorm_func(y, **denorm_kwargs)
