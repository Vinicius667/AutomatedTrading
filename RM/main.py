import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.model_selection import GridSearchCV
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

from callbacks import early_stop_callback, print_callback
from data_module import CustomDataModule
from model import LSTMModel

pl.seed_everything(42)
model = LSTMModel(hidden_size=150, num_layers=2)
trainer = pl.Trainer(min_epochs=1, max_epochs=250, callbacks=[early_stop_callback, print_callback], check_val_every_n_epoch=10)
dm = CustomDataModule(num_data_points=10, batch_size=20)
trainer.fit(model, dm)
