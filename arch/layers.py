import torch.nn as nn 
import torch 
import torch.nn.functional as F
import math

class NormLinear(nn.Module):
    def __init__(
        self, in_features, out_features, temperature=0.05, temperature_trainable=False
    ):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.scale = 1 / temperature
        if temperature_trainable:
            self.scale = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.scale, 1 / temperature)

    def forward(self, x):
        x_norm = F.normalize(x)
        w_norm = F.normalize(self.weight)
        cosine = F.linear(x_norm, w_norm, None)
        out = cosine  # * self.scale
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class ArcMarginProductSubcenter(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine