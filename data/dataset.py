import torch
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from augs.augmentations import get_test_transforms, get_train_transforms
import ast
from PIL import Image
import numpy as np
import pandas as pd


class WhaleDataset(torch.nn.Module):
    def __init__(self, params, df, transforms, stage):
        self.path = params.path
        self.stage = stage
        self.transforms = transforms
        self.params = params
        self.crops = pd.read_csv(params.crop_path)
        # frame = pd.read_csv(params.csv_path)
        self.image_ids = df.Image.values.tolist()
        self.counts = df.label.value_counts().sort_index().values
        self.targets = df.label.values.tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, indx):
        image_idx = self.image_ids[indx]
        image_path = f"{self.path}{image_idx}"
        if self.stage == "train" or self.stage == "val":
            bb = self.crops[self.crops.paths == image_idx]
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            bboxes = [
                int(bb.x0.values[0] * width),
                int(bb.y0.values[0] * height),
                int(bb.x1.values[0] * width),
                int(bb.y1.values[0] * height),
            ]
            #print(bboxes)
            # if isinstance(crops, list):
            img = img.crop(bboxes)
            targets = self.targets[indx]
        else:
            img = Image.open(image_path)
        img = np.array(img)
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]

        if self.stage == "train" or self.stage == "val":
            return img, targets, indx
        else:
            return img, image_idx, indx


class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.
    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0, replacement=True):
        # if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool):
        # raise ValueError("num_samples should be a non-negative integeral "
        # "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [weights[i] for i in self.indices]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, self.replacement)
        )

    def __len__(self):
        return self.num_samples


def getDataLoader(df, params, stage, train_weights, replacement):
    if stage == "train":
        trnsfms = get_train_transforms(params)
    else:
        trnsfms = get_test_transforms(params)

    dataset = WhaleDataset(params, df, trnsfms, stage)
    if stage == "train":
        tr_ind = np.arange(0, len(dataset), 1)
        train_sampler = WeightedSubsetRandomSampler(
            tr_ind, train_weights, replacement=replacement
        )
        loader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        # print("Here")
    else:
        loader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=True,
            shuffle=False,
        )
    return loader
    #return dataset
