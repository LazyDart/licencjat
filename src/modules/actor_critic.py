from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        hidden_dim: int,
        continuous_action_space: bool = False,
        action_std_init: float = 0.0,
        device: str | torch.device = "cpu",
        feature_extractor_override: nn.Module | None = None,
    ):
        super(ActorCritic, self).__init__()
        self.continuous_action_space = continuous_action_space
        self.device = device

        # create shared feature extractor for both actor and critic
        if feature_extractor_override is None:
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_dim[0], hidden_dim, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
                nn.Tanh(),
            ).to(device)
        else:
            self.feature_extractor = deepcopy(feature_extractor_override).to(device)

        if continuous_action_space:
            self.action_var = nn.Parameter(
                torch.full(size=(action_dim,), fill_value=action_std_init * action_std_init)
            ).to(device)
            self.actor_head = nn.Linear(hidden_dim, action_dim, dtype=torch.float32).to(device)
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim, dtype=torch.float32), nn.Softmax(dim=-1)
            ).to(device)

        self.critic_head = nn.Linear(hidden_dim, 1).to(device)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)
        return actor_out, critic_out

    def select_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Ensure tensor, float32, and add batch dim
        obs = obs.to(torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_out, value = self.forward(obs)  # action_out: logits or means
            if self.continuous_action_space:
                action_cov = torch.diag(self.action_var)
                dist = MultivariateNormal(action_out, action_cov)
            else:
                dist = Categorical(logits=action_out)

            sampled = dist.sample()  # [1, A] or [1]
            logprob = dist.log_prob(sampled)  # [1] or [1]

        # Squeeze batch, move to CPU for the buffer
        if self.continuous_action_space:
            act_tensor = sampled.squeeze(0).to("cpu").to(torch.float32)  # [A]
            # env_action = act_tensor.numpy()                                  # np.ndarray for env
        else:
            act_tensor = sampled.squeeze(0).to("cpu").to(torch.int64)  # [] int64
            # env_action = int(act_tensor.item())                              # int for env

        return act_tensor, logprob.squeeze(0).to("cpu"), value.squeeze(0).to("cpu")

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_out, values = self.forward(states)

        if self.continuous_action_space:
            action_cov = torch.diag(self.action_var)
            dist = MultivariateNormal(action_out, action_cov)
            action_logprobs = dist.log_prob(actions)
        else:
            dist = Categorical(action_out)
            action_logprobs = dist.log_prob(actions.squeeze(-1).long())
        dist_entropy = dist.entropy()

        return values.squeeze(), action_logprobs, dist_entropy
