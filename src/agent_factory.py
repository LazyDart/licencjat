from torch import nn

from src.buffer import TensorRolloutBuffer
from src.config import Config
from src.modules.feature_extractors import MinigridFeaturesExtractor, MLPFeatureExtractor
from src.ppo import PPOAgent, PPOAgentICM


class AgentFactory:
    def __init__(
        self,
        obs_dim: tuple[int, ...],
        action_dim: int,
        config: Config,
        device: str,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

    def _make_feature_extractor(self) -> nn.Module:
        if "minigrid" in self.config.env_name.lower() or "babyai" in self.config.env_name.lower():
            feature_extractor = MinigridFeaturesExtractor(self.obs_dim, self.config.hidden_dim)
        else:
            feature_extractor = MLPFeatureExtractor(self.obs_dim, self.config.hidden_dim)

        return feature_extractor

    def _make_buffer(self) -> TensorRolloutBuffer:
        store_next = self.config.agent == "ppo_icm"
        horizon = self.config.update_interval  # PPO-style; could also be config.max_eps_steps
        action_shape = (self.action_dim,) if self.config.continuous_action_space else None

        buffer = TensorRolloutBuffer(
            horizon=horizon,
            obs_shape=self.obs_dim,
            action_shape=action_shape,
            discrete_actions=not self.config.continuous_action_space,
            store_next_state=store_next,
            pin_memory=True,
        )
        return buffer

    def _make_ppo_agent(
        self, buffer: TensorRolloutBuffer, feature_extractor: nn.Module
    ) -> PPOAgent:
        ppo_agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            buffer=buffer,
            feature_extractor=feature_extractor,
            continuous_action_space=self.config.continuous_action_space,
            num_epochs=self.config.num_epochs,
            eps_clip=self.config.eps_clip,
            action_std_init=self.config.action_std_init,
            gamma=self.config.gamma,
            entropy_coef=self.config.entropy_coef,
            value_loss_coef=self.config.value_loss_coef,
            batch_size=self.config.batch_size,
            max_grad_norm=self.config.max_grad_norm,
            device=self.device,
        )
        return ppo_agent

    def _make_ppo_icm_agent(
        self, buffer: TensorRolloutBuffer, feature_extractor: nn.Module
    ) -> PPOAgentICM:
        assert (
            isinstance(self.config.icm_beta, float)
            and isinstance(self.config.icm_eta, float)
            and isinstance(self.config.lr_icm, float)
        ), "Missing required args!"
        assert buffer.next_states is not None, "Buffer next_state must be not None in ICM."

        ppo_agent = PPOAgentICM(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            lr_actor=self.config.lr_actor,
            lr_critic=self.config.lr_critic,
            buffer=buffer,
            feature_extractor=feature_extractor,
            continuous_action_space=self.config.continuous_action_space,
            num_epochs=self.config.num_epochs,
            eps_clip=self.config.eps_clip,
            action_std_init=self.config.action_std_init,
            gamma=self.config.gamma,
            entropy_coef=self.config.entropy_coef,
            value_loss_coef=self.config.value_loss_coef,
            batch_size=self.config.batch_size,
            max_grad_norm=self.config.max_grad_norm,
            device=self.device,
            icm_beta=self.config.icm_beta,
            icm_eta=self.config.icm_eta,
            lr_icm=self.config.lr_icm,
        )
        return ppo_agent

    def make_agent(self) -> PPOAgent:
        buffer = self._make_buffer()
        feature_extractor = self._make_feature_extractor()

        if self.config.agent.lower() == "ppo":
            agent = self._make_ppo_agent(buffer, feature_extractor)

        elif self.config.agent.lower() == "ppo_icm":
            agent = self._make_ppo_icm_agent(buffer, feature_extractor)
        else:
            raise ValueError(f"Invalid agent name: {self.config.agent}")

        return agent
