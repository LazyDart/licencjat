from copy import deepcopy

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from src.buffer import TensorRolloutBuffer
from src.modules.actor_critic import ActorCritic
from src.modules.icm import ICM, icm_training_step
from src.modules.random_network_distillation import RandomNetworkDistillation, rnd_training_step


class PPOAgent:
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        hidden_dim: int,
        lr_actor: float,
        lr_critic: float,
        buffer: TensorRolloutBuffer,
        feature_extractor: nn.Module,
        continuous_action_space: bool = False,
        num_epochs: int = 10,
        eps_clip: float = 0.2,
        action_std_init: float = 0.6,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_std_init = action_std_init
        self.continuous_action_space = continuous_action_space
        self.device = device

        self.policy = ActorCritic(
            action_dim,
            hidden_dim,
            deepcopy(feature_extractor).to(device),
            continuous_action_space=continuous_action_space,
            action_std_init=action_std_init,
            device=device,
        )

        self.buffer = buffer
        self.mse_loss = nn.MSELoss()  # Initialize MSE loss

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.feature_extractor.parameters()},
                {"params": self.policy.actor_head.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_head.parameters(), "lr": lr_critic},
            ]
        )

    def compute_returns(self) -> torch.Tensor:
        returns = torch.empty_like(self.buffer.rewards)
        discounted_reward: float = 0

        for i, (reward, done) in enumerate(
            zip(reversed(self.buffer.rewards), reversed(self.buffer.dones), strict=False), start=1
        ):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns[-i] = discounted_reward

        returns = returns.to(self.device)
        return returns

    def _update_policy_with_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        rewards_to_go: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # evaluate old actions and values
        state_values, logprobs, dist_entropy = self.policy.evaluate_actions(states, actions)

        # Finding the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(logprobs - old_logprobs.squeeze(-1))

        # Finding Surrogate Loss

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # final loss of clipped objective PPO
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), rewards_to_go)
        loss = (
            actor_loss
            + self.value_loss_coef * critic_loss
            - self.entropy_coef * dist_entropy.mean()
        )

        # calculate gradients and backpropagate for actor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def update_weights(self) -> None:
        rewards_to_go = self.compute_returns()

        states = self.buffer.states.to(self.device)
        actions = self.buffer.actions.to(self.device)
        old_logprobs = self.buffer.logprobs.to(self.device)
        state_vals = self.buffer.state_values.to(self.device)

        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.num_epochs):
            # generate random indices for minibatch
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]

                self._update_policy_with_batch(
                    states=batch_states,
                    actions=batch_actions,
                    old_logprobs=batch_old_logprobs,
                    rewards_to_go=batch_rewards_to_go,
                    advantages=batch_advantages,
                )

        self.buffer.clear()


class PPOAgentICM(PPOAgent):
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        hidden_dim: int,
        lr_actor: float,
        lr_critic: float,
        lr_icm: float,
        buffer: TensorRolloutBuffer,
        feature_extractor: nn.Module,
        continuous_action_space: bool = False,
        num_epochs: int = 10,
        eps_clip: float = 0.2,
        action_std_init: float = 0.6,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        icm_beta: float = 0.2,
        intrinsic_reward_eta: float = 0.01,
        skip_interval: float | None = None,
        skip_prob: float | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        continuous_action_space = False  # ICM does not support discountinous actions yet.

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            buffer=buffer,
            feature_extractor=feature_extractor,
            continuous_action_space=continuous_action_space,
            num_epochs=num_epochs,
            eps_clip=eps_clip,
            action_std_init=action_std_init,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        self.icm_beta = icm_beta
        self.int_rew_eta = intrinsic_reward_eta
        self.int_rew_model = ICM(
            deepcopy(feature_extractor), action_dim, hidden_dim, hidden_dim
        ).to(device)
        self.skip_interval = skip_interval
        self.skip_prob = skip_prob
        self.stop_updates = False
        self.n_updates = 0

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.feature_extractor.parameters()},
                {"params": self.policy.actor_head.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_head.parameters(), "lr": lr_critic},
                {"params": self.int_rew_model.head.parameters(), "lr": lr_icm},
                {"params": self.int_rew_model.inverse_model_network.parameters(), "lr": lr_icm},
                {"params": self.int_rew_model.next_state_pred_network.parameters(), "lr": lr_icm},
            ]
        )
        self.buffer = buffer

    def _update_icm_with_batch(
        self,
        batch_states: torch.Tensor,
        batch_next_states: torch.Tensor,
        batch_actions: torch.Tensor,
    ) -> None:
        td = TensorDict(
            {
                "action": batch_actions.long(),
                "state": batch_states,
                "next_state": batch_next_states,
            }
        )
        icm_training_step(
            self.int_rew_model,
            self.optimizer,
            td,
            self.icm_beta,
            self.int_rew_eta,
            device=self.device,
        )

    def update_weights(self) -> None:
        self.n_updates += 1
        next_states = self.buffer.next_states.to(self.device)
        states = self.buffer.states.to(self.device)
        actions = self.buffer.actions.to(self.device)
        old_logprobs = self.buffer.logprobs.to(self.device)
        state_vals = self.buffer.state_values.to(self.device)

        rewards_to_go = self.compute_returns()
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.num_epochs):
            # generate random indices for minibatch
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]

                self._update_policy_with_batch(
                    states=batch_states,
                    actions=batch_actions,
                    old_logprobs=batch_old_logprobs,
                    rewards_to_go=batch_rewards_to_go,
                    advantages=batch_advantages,
                )

                if (
                    (self.skip_interval is None or (self.n_updates % self.skip_interval != 0))
                    and (not self.stop_updates)
                    and (self.skip_prob is None or (np.random.rand() > self.skip_prob))
                ):
                    self._update_icm_with_batch(batch_states, batch_next_states, batch_actions)

        self.buffer.clear()


class PPOAgentRND(PPOAgent):
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        hidden_dim: int,
        lr_actor: float,
        lr_critic: float,
        lr_rnd: float,
        buffer: TensorRolloutBuffer,
        feature_extractor: nn.Module,
        continuous_action_space: bool = False,
        num_epochs: int = 10,
        eps_clip: float = 0.2,
        action_std_init: float = 0.6,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        intrinsic_reward_eta: float = 0.01,
        skip_interval: float | None = None,
        skip_prob: float | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            buffer=buffer,
            feature_extractor=feature_extractor,
            continuous_action_space=continuous_action_space,  # RND works for both
            num_epochs=num_epochs,
            eps_clip=eps_clip,
            action_std_init=action_std_init,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        self.int_rew_eta = intrinsic_reward_eta
        self.skip_interval = skip_interval
        self.skip_prob = skip_prob
        self.stop_updates = False
        self.n_updates = 0

        # As with your ICM init, assume feature_extractor outputs `hidden_dim` features.
        # Use a deepcopy so policy and RND have separate encoders.
        self.int_rew_model = RandomNetworkDistillation(
            head=deepcopy(feature_extractor),
            feature_dim=hidden_dim,
            hidden=hidden_dim,
        ).to(device)

        # Optimizer: include policy + RND params (predictor + encoder). Target is frozen.
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.feature_extractor.parameters()},
                {"params": self.policy.actor_head.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_head.parameters(), "lr": lr_critic},
                {"params": self.int_rew_model.predictor.parameters(), "lr": lr_rnd},
            ]
        )
        self.buffer = buffer

    def _update_rnd_with_batch(
        self,
        batch_states: torch.Tensor,
        batch_next_states: torch.Tensor,
    ) -> None:
        td = TensorDict(
            {
                "state": batch_states,
                "next_state": batch_next_states,  # RND uses next_state novelty by default
            }
        )
        rnd_training_step(
            self.int_rew_model, self.optimizer, td, eta=self.int_rew_eta, device=self.device
        )

    def update_weights(self) -> None:
        self.n_updates += 1

        next_states = self.buffer.next_states.to(self.device)
        states = self.buffer.states.to(self.device)
        actions = self.buffer.actions.to(self.device)
        old_logprobs = self.buffer.logprobs.to(self.device)
        state_vals = self.buffer.state_values.to(self.device)

        rewards_to_go = self.compute_returns()
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        for _ in range(self.num_epochs):
            indices = np.random.permutation(len(self.buffer.states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards_to_go = rewards_to_go[batch_indices]

                # PPO policy/critic update
                self._update_policy_with_batch(
                    states=batch_states,
                    actions=batch_actions,
                    old_logprobs=batch_old_logprobs,
                    rewards_to_go=batch_rewards_to_go,
                    advantages=batch_advantages,
                )

                # Optional RND update (mirrors your ICM skip logic)
                if (
                    (self.skip_interval is None or (self.n_updates % self.skip_interval != 0))
                    and (not self.stop_updates)
                    and (self.skip_prob is None or (np.random.rand() > self.skip_prob))
                ):
                    self._update_rnd_with_batch(batch_states, batch_next_states)

        self.buffer.clear()
