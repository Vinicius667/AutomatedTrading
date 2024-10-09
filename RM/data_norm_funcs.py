import numpy as np
import pandas as pd
import torch


def normalize_sub_mean_min_max(x: pd.Series) -> tuple[torch.Tensor, dict[str, float]]:
    # Normalize the data version 1

    target_mean = x.mean()
    x = x - target_mean

    target_max = x.max()
    target_min = x.min()

    return (x - target_min) / (target_max - target_min), {"max": target_max, "min": target_min, "mean": target_mean}


def denormalize_sub_mean_min_max(y: torch.Tensor, **kwargs: float) -> torch.Tensor:
    # Reverse the normalization version 1

    target_mean = kwargs["mean"]
    target_max = kwargs["max"]
    target_min = kwargs["min"]
    return y * (target_max - target_min) + (target_min) + target_mean


dict_norm_denorm_func_pairs = {
    "sub_mean_min_max": (normalize_sub_mean_min_max, denormalize_sub_mean_min_max),
}


if __name__ == "__main__":
    x = pd.Series(np.random.rand(10000))
    for norm_name, (norm_func, denorm_func) in dict_norm_denorm_func_pairs.items():
        x_norm, kwargs = norm_func(x)
        x_denorm = denorm_func(x_norm, **kwargs)
        print(f"{norm_name}: Error: {(x - x_denorm).abs().max()}")
