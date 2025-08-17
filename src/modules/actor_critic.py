from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        continuous_action_space=False,
        action_std_init=0.0,
        device="cpu",
        feature_extractor_override: nn.Module = None,
    ):
        super(ActorCritic, self).__init__()
        self.continuous_action_space = continuous_action_space
        self.device = device

        # create shared feature extractor for both actor and critic
        if feature_extractor_override is None:
            self.feature_extractor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim, dtype=torch.float32),
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

    def forward(self, obs):
        features = self.feature_extractor(obs)
        actor_out = self.actor_head(features)
        critic_out = self.critic_head(features)
        return actor_out, critic_out

    def select_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        # print("Observation dim:", obs.dim())
        obs = obs.unsqueeze(0)  # add batch dimension if missing

        # to prevent unnecessary gradient computation
        with torch.no_grad():
            action_out, value = self.forward(obs)
            # print('stage-0:', action_out.shape, value, obs.shape)

            if self.continuous_action_space:
                action_cov = torch.diag(self.action_var)  # (na, na)
                # print('stage-1:', action_out.shape, action_cov.shape)
                dist = MultivariateNormal(action_out, action_cov)
            else:
                # print(action_out.shape)
                dist = Categorical(action_out)

            action = dist.sample()
            action_logprob = dist.log_prob(action)

            if self.continuous_action_space:
                if action.dim() == 2 and action.shape[0] == 1:
                    action = action.squeeze(0).cpu().numpy()
            else:
                # action = torch.clamp(action, -1.0, 1.0)
                action = action.item()

        return action, action_logprob.cpu().numpy(), value.item()

    def evaluate_actions(self, states, actions):
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
