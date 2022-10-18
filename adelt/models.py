import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, dim_in, dim, n_classes_pytorch, n_classes_keras, drop):
        super().__init__()
        self.in_transform = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim_in, dim),
            nn.ReLU()
        )
        self.classifier_pytorch = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim, n_classes_pytorch)
        )
        self.classifier_keras = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim, n_classes_keras)
        )

    def forward(self, x, mask, run_classifier):
        """
        :param x: Float[bs * 2, max_len, dim]
        :type x: torch.Tensor
        :param mask: Float[bs * 2, max_len]; 1.0 if ok, 0.0 if padding
        :type mask: torch.Tensor
        :param run_classifier:
        :type run_classifier: bool
        :rtype: tuple
        """
        x = self.in_transform(x)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        bs = x.size(0) // 2
        assert x.size(0) == bs * 2
        x1, x2 = x[:bs, :], x[bs:, :]
        if run_classifier:
            return x1, x2, self.classifier_pytorch(x1), self.classifier_keras(x2)
        else:
            return x1, x2


class Discriminator(nn.Module):
    def __init__(self, dim, dim_hid, drop, leaky):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(dim, dim_hid),
            nn.LeakyReLU(leaky),
            nn.Dropout(drop),
            nn.Linear(dim_hid, 1)
        )

    def forward(self, x):
        """
        :param x: Float[bs * 2, dim]
        :type x: torch.Tensor
        :returns: Float[bs * 2]
        :rtype: torch.Tensor
        """
        return self.layers(x).squeeze(-1)
