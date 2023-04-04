import os
import random
import torch
import numpy as np
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import json
from dotenv import load_dotenv
from pathlib import Path
import mlflow
import argparse
from datetime import datetime
import math
load_dotenv()




def stdout_info(logger, params):
    current_timestamp = datetime.now()
    current_timestamp = current_timestamp.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"Date: {current_timestamp}")
    infoStr = f"Run started: {params.backbone}-{params.loss_name}"
    logger.info(infoStr)
    logger.info("---" * 20)


def format_string(data) -> str:
    outputString = ""
    for k, v in data.items():
        outputString += f"{k} : {v:.2f} | "
    return outputString



def stdout_results(logger, metricList, current_epoch, epochs, time):
    inputstr = f"Epoch : {current_epoch}/{epochs} | Time: {asMinutes(time)} | " 
    for metrics in metricList:
        inputstr += format_string(metrics)
    logger.info(inputstr)
    logger.info("---" * 20)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    return parser


def set_mlflow_vars():
    registry_path = os.getenv("MODEL_REGISTRY")
    registry_path = Path(registry_path)
    Path(registry_path).mkdir(exist_ok=True) # create experiments dir
    mlflow.set_tracking_uri("file://" + str(registry_path.absolute()))


def _store_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent = 4)


def check_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    #return path

def get_logger(filename='./logs/train'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class EarlyStopping:
    def __init__(self, mode, patience):
        self.mode = mode
        self._best_score = float("inf") if self.mode == "min" else -float("inf")
        self._es_counter = 0
        self._patience = patience
        self.early_stop = False

    def _improvement(self, score):
        if self.mode == "min":
            return score <= self._best_score
        else:
            return score >= self._best_score

    def step(self, score):
        if self._improvement(score):
            self._best_score = score
            self._es_counter = 0
        else:
            self._es_counter += 1
            if self._es_counter >= self._patience:
                print("Early Stopping !")
                self.early_stop = True


class ModelCheckpointer:
    def __init__(self, mode, path):
        self.mode = mode
        self._best_score = float("inf") if self.mode == "min" else -float("inf")
        self.path = path

    def _improvement(self, score):
        if self.mode == "min":
            return score <= self._best_score
        else:
            return score >= self._best_score

    def step(self, score, model_state):
        if self._improvement(score):
            self._best_score = score
            torch.save(model_state, self.path)
