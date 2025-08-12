import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from tensordict import TensorDict

from icm import ICM, icm_training_step


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    # TODO: perhaps modify the code so that state pairs: prev/next states are kept.
    def store_transition(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()


class RolloutBufferNextState(RolloutBuffer):
    def __init__(self):
        super().__init__()
        self.next_states = []

    def store_transition(self, state, next_state, action, logprob, reward, done, state_value):
        super().store_transition(state, action, logprob, reward, done, state_value)
        self.next_states.append(next_state)

    def clear(self):
        super().clear()
        self.next_states.clear()

class ActorCritic(nn.Module):
    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            hidden_dim, 
            continuous_action_space=False, 
            action_std_init=0.0, 
            device='cpu'
        ):
        super(ActorCritic, self).__init__()
        self.continuous_action_space = continuous_action_space
        self.device = device

        # create shared feature extractor for both actor and critic
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh()
        ).to(device)

        if continuous_action_space:
            self.action_var = nn.Parameter(torch.full(size=(action_dim,), fill_value=action_std_init * action_std_init)).to(device)
            self.actor_head = nn.Linear(hidden_dim, action_dim, dtype=torch.float32).to(device)
        else:
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim, action_dim, dtype=torch.float32),
                nn.Softmax(dim=-1)
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
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)    # add batch dimension if missing

        # to prevent unnecessary gradient computation
        with torch.no_grad():
            action_out, value = self.forward(obs)
            # print('stage-0:', action_out.shape, value, obs.shape)

            if self.continuous_action_space:
                action_cov = torch.diag(self.action_var)    # (na, na)
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


class PPOAgent:
    def __init__(
            self, 
            obs_dim, 
            action_dim, 
            hidden_dim, 
            lr_actor, 
            lr_critic, 
            continuous_action_space=False, 
            num_epochs=10, 
            eps_clip=0.2, 
            action_std_init=0.6, 
            gamma=0.99,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            batch_size=64,
            max_grad_norm=0.5,
            device='cpu'
        ):
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
        )

        self.mse_loss = nn.MSELoss()  # Initialize MSE loss


    def compute_returns(self):
        returns = []
        discounted_reward = 0

        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = np.array(returns, dtype=np.float32)
        returns = torch.flatten(torch.from_numpy(returns).float()).to(self.device)
        return returns


    def _update_policy_with_batch(self, states, actions, old_logprobs, rewards_to_go, advantages):
        
        # evaluate old actions and values
        state_values, logprobs, dist_entropy = self.policy.evaluate_actions(states, actions)
        # print(logprobs.shape, batch_old_logprobs.shape)

        # Finding the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(logprobs - old_logprobs.squeeze(-1))

        # Finding Surrogate Loss
        # print(ratios.shape, batch_advantages.shape)
        surr1 = ratios * advantages 
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

        # final loss of clipped objective PPO
        actor_loss = -torch.min(surr1, surr2).mean()
        # print(state_values.dtype, batch_rewards_to_go.dtype)
        critic_loss = 0.5 * self.mse_loss(state_values.squeeze(), rewards_to_go)
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
        # print("Final loss:", actor_loss, critic_loss, dist_entropy, loss)

        # calculate gradients and backpropagate for actor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_weights(self):
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(self.device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(self.device)

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

    def __init__(self,
            obs_dim, 
            action_dim, 
            hidden_dim, 
            lr_actor, 
            lr_critic,
            lr_icm,
            continuous_action_space=False, 
            num_epochs=10, 
            eps_clip=0.2, 
            action_std_init=0.6, 
            gamma=0.99,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            batch_size=64,
            max_grad_norm=0.5,
            icm_beta=0.2,
            icm_eta=0.01,
            device='cpu'
        ):
        
        super().__init__(
            self, 
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            hidden_dim=hidden_dim, 
            lr_actor=lr_actor, 
            lr_critic=lr_critic, 
            continuous_action_space=continuous_action_space, 
            num_epochs=num_epochs, 
            eps_clip=eps_clip, 
            action_std_init=action_std_init, 
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            device=device
        )
        icm_head = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.Tanh()
        ).to(device)

        self.icm_beta = icm_beta
        self.icm_eta = icm_eta
        self.icm = ICM(icm_head, action_dim, obs_dim, hidden_dim)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.feature_extractor.parameters()},
            {'params': self.policy.actor_head.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic},
            {'params': self.icm.head.parameters(), 'lr': lr_icm},
            {'params': self.icm.inverse_model_network.parameters(), 'lr': lr_icm},
            {'params': self.icm.next_state_pred_network.parameters(), 'lr': lr_icm},
        ])

        self.buffer = RolloutBufferNextState()
        
        self.buffer.clear()

    def update_icm(self):
        # print(len(self.buffer.rewards))
        rewards_to_go = self.compute_returns()
        # print(len(rewards_to_go))

        next_states = torch.from_numpy(np.array(self.buffer.next_states)).float().to(self.device)
        states = torch.from_numpy(np.array(self.buffer.states)).float().to(self.device)
        actions = torch.from_numpy(np.array(self.buffer.actions)).float().to(self.device)
        old_logprobs = torch.from_numpy(np.array(self.buffer.logprobs)).float().to(self.device)
        state_vals = torch.from_numpy(np.array(self.buffer.state_values)).float().to(self.device)

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

                td = TensorDict({
                    "action": batch_actions,
                    "state": batch_states,
                    "next_state": batch_next_states,
                })
                icm_training_step(self.icm, self.optimizer, td, self.icm_beta, self.icm_eta, device=self.device)
        

    def update_weights(self):
        self._update_policy()



        self.buffer.clear()

