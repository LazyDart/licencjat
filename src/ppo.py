from copy import deepcopy

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from src.buffer import TensorRolloutBuffer
from src.modules.actor_critic import ActorCritic
from src.modules.icm import ICM, icm_training_step


class PPOAgent:
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        hidden_dim: int,
        lr_actor: float,
        lr_critic: float,
        buffer: TensorRolloutBuffer,
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
        feature_extractor_override: nn.Module | None = None,
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
            obs_dim,
            action_dim,
            hidden_dim,
            continuous_action_space=continuous_action_space,
            action_std_init=action_std_init,
            device=device,
            feature_extractor_override=feature_extractor_override,
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
        returns: list[float] = []
        discounted_reward: float = 0

        for reward, done in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.dones), strict=False
        ):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = np.array(returns, dtype=np.float32)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
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
        # print(logprobs.shape, batch_old_logprobs.shape)

        # Finding the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(logprobs - old_logprobs.squeeze(-1))

        # Finding Surrogate Loss
        # print(ratios.shape, batch_advantages.shape)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        # final loss of clipped objective PPO
        actor_loss = -torch.min(surr1, surr2).mean()
        # print(state_values.dtype, batch_rewards_to_go.dtype)
        critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), rewards_to_go)
        loss = (
            actor_loss
            + self.value_loss_coef * critic_loss
            - self.entropy_coef * dist_entropy.mean()
        )
        # print("Final loss:", actor_loss, critic_loss, dist_entropy, loss)

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
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(self.device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(self.device)

        # states = self.buffer.states.to(self.device)
        # actions = self.buffer.actions.to(self.device)
        # old_logprobs = self.buffer.logprobs.to(self.device)
        # state_vals = self.buffer.state_values.to(self.device)
        
        # print('stage-0:', rewards_to_go.shape, state_vals.shape)
        # print('stage-1:', rewards_to_go.device, state_vals.device)
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # print(states.shape, actions.shape, old_logprobs.shape, state_vals.shape, advantages.shape, rewards_to_go.shape)

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
        icm_eta: float = 0.01,
        device: str | torch.device = "cpu",
        feature_extractor_override: nn.Module | None = None,
    ) -> None:
        continuous_action_space = False  # ICM does not support discountinous actions yet.

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            buffer=buffer,
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
            feature_extractor_override=feature_extractor_override,
        )
        if feature_extractor_override is None:
            icm_head = nn.Sequential(
                nn.Linear(obs_dim[0], hidden_dim, dtype=torch.float32),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
                nn.Tanh(),
            ).to(device)
        else:
            icm_head = deepcopy(feature_extractor_override).to(device)

        self.icm_beta = icm_beta
        self.icm_eta = icm_eta
        self.icm = ICM(icm_head, action_dim, hidden_dim, hidden_dim).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.feature_extractor.parameters()},
                {"params": self.policy.actor_head.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_head.parameters(), "lr": lr_critic},
                {"params": self.icm.head.parameters(), "lr": lr_icm},
                {"params": self.icm.inverse_model_network.parameters(), "lr": lr_icm},
                {"params": self.icm.next_state_pred_network.parameters(), "lr": lr_icm},
            ]
        )

        self.buffer = buffer

    def update_icm(self) -> None:
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        next_states = torch.from_numpy(np.array(self.buffer.next_states)).float().to(self.device)
        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(self.device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(self.device)

        # next_states = self.buffer.next_states.to(self.device)
        # states = self.buffer.states.to(self.device)
        # actions = self.buffer.actions.to(self.device)
        # old_logprobs = self.buffer.logprobs.to(self.device)
        # state_vals = self.buffer.state_values.to(self.device)

        # print('stage-0:', rewards_to_go.shape, state_vals.shape)
        # print('stage-1:', rewards_to_go.device, state_vals.device)
        advantages = rewards_to_go - state_vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # print(states.shape, actions.shape, old_logprobs.shape, state_vals.shape, advantages.shape, rewards_to_go.shape)

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
                td = TensorDict(
                    {
                        "action": batch_actions.long(),
                        "state": batch_states,
                        "next_state": batch_next_states,
                    }
                )
                icm_training_step(
                    self.icm, self.optimizer, td, self.icm_beta, self.icm_eta, device=self.device
                )

    def update_weights(self) -> None:
        self.update_icm()

        self.buffer.clear()
