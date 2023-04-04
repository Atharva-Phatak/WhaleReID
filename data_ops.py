from data.dataset import getDataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from dagster import In, Out, op
from omegaconf import DictConfig
import torch
from typing import Dict
from omegaconf import OmegaConf, DictConfig


@op(config_schema={"path": str})
def get_params(context) -> DictConfig:
    """Get parameters for training from config file"""
    return OmegaConf.load(context.op_config["path"])


@op(ins={"params": In(DictConfig), "stage": In(str), "replacement": In(bool)})
def get_train_dl(params, stage, replacement):
    dataframe = pd.read_csv(params.train_data_path)
    sample_weights = np.ones(dataframe.shape[0])
    return getDataLoader(dataframe, params, stage, sample_weights, replacement)


@op(ins={"params": In(DictConfig), "stage": In(str)})
def get_val_dl(params, stage):
    dataframe = pd.read_csv(params.val_data_path)
    return getDataLoader(
        dataframe, params, stage=stage, train_weights=None, replacement=False
    )


@op(
    ins={"params": In(DictConfig)},
    out={"dataloaders": Out(Dict[str, torch.utils.data.DataLoader])},
)
def get_dataloaders(params):
    """Method to create dataloaders"""
    # data = prep_data(params)
    train_dl = get_train_dl(params=params, stage="train", replacement=False)
    val_dl = get_val_dl(params=params, stage="val")
    dataloaders = {"train": train_dl, "validation": val_dl}
    return dataloaders
