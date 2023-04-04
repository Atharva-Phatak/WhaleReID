from dataclasses import dataclass
from typing import Any


@dataclass
class Params:
    csv_path: str = "../data/train.csv"
    train_data_path: str = "./csv_store/train_data.csv"
    val_data_path: str = "./csv_store/val_data.csv"
    crop_path: str = "../data/yolo_preds_v2.csv"
    num_workers: int = 0
    epochs: int = 1
    batch_size: int = 16
    image_size: int = 448
    last_linear_lr: float = 1.6e-2
    lr: float = 1.6e-3
    backbone: str = "resnet18"
    seed: int = 42
    path: str = "../data/train/"
    device: str = "cuda"
    save_path: str = "model.pt"
    es_patience: int = 5
    embedding_size: Any = None
    warmup_steps_ratio: float = 0.2
    lr_decay_scale: float = 1.0e-2
