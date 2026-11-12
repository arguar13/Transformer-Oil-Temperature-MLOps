"""DLinear architecture: the single definition inside `core_ml`.

train.py, batch_inference.py and mlflow_utils.py import the network from here.
The module is also copied as is into the API image (see `Dockerfile`), because
unpickling the registered model rebuilds this class by module reference
(`src.model_architecture.DLinear`) in whatever container loads it.
"""

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023) with a univariate output head.

    1. Decompose every input channel into trend (moving average) + seasonal
       part (residual).
    2. Project each part with a seq_len -> pred_len linear layer shared across
       channels: every horizon step has its own weights over the past seq_len
       readings, as in the paper.
    3. Mix the n_features channels into the single target (OT), step by step.

    Input: [batch, seq_len, n_features]. Output: [batch, pred_len].
    """

    def __init__(self, seq_len=48, n_features=10, pred_len=1, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = series_decomp(kernel_size)
        # seq_len -> pred_len, not seq_len -> 1: projecting to one scalar per
        # channel would make all pred_len steps come from the same n_features
        # numbers (10 values to predict 48 hours) instead of having their own
        # temporal weights.
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        self.combine = nn.Linear(n_features, 1)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # [batch, n_features, pred_len]
        x = self.Linear_Seasonal(seasonal_init) + self.Linear_Trend(trend_init)

        # [batch, pred_len, n_features] -> [batch, pred_len, 1] -> [batch, pred_len]
        return self.combine(x.permute(0, 2, 1)).squeeze(-1)
