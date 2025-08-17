import torch
from torch import nn


class MinigridFeaturesExtractor(nn.Module):
    def __init__(self, obs_dim: tuple[int, int, int], features_dim: int = 512) -> None:
        super().__init__()
        n_input_channels = obs_dim[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            x = torch.zeros(1, *obs_dim).permute((0, 3, 1, 2))
            y = self.cnn(x)
            linear_layer_input_shape = y.shape[1]

        self.linear = nn.Sequential(nn.Linear(linear_layer_input_shape, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float().permute((0, 3, 1, 2))))
