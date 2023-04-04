import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math


class CosineMarginCrossEntropy(nn.Module):
    def __init__(self, m=0.35, s=32.0):
        super(CosineMarginCrossEntropy, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (input - one_hot * self.m)

        loss = self.ce(output, target)
        return loss


class ArcMarginCrossEntropy(nn.Module):
    def __init__(self, m=0.50, s=30.0, m_cos=0.3):
        super(ArcMarginCrossEntropy, self).__init__()
        self.m = m
        self.m_cos = m_cos
        self.s = s
        self.ce = nn.CrossEntropyLoss()

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, target):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        # output = output - one_hot * self.m_cos # cosine-margin
        output *= self.s

        loss = self.ce(output, target)
        return loss


class ArcFaceLossAdaptiveMargin(nn.modules.Module):
    def __init__(self, margins: np.ndarray, n_classes: int, s: float = 30.0):
        super().__init__()
        self.s = s
        self.margins = margins
        self.out_dim = n_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        outputs = ((labels * phi) + ((1.0 - labels) * cosine)) * self.s
        return self.ce(outputs, labels)