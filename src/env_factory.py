import gymnasium as gym
import minigrid

from src.config import Config


class EnvFactory:
    def __init__(self, config: Config) -> None:
        self.config = config

    def create_env(self) -> gym.Env:
        env_name = self.config.env_name
        try:
            env = gym.make(env_name)
            if "minigrid" in env_name.lower():
                env = minigrid.wrappers.ImgObsWrapper(env)
        except:
            raise ValueError(f"Invalid environment name: {env_name}")
        return env

    def get_env_dims(self, env: gym.Env) -> tuple[tuple[int, ...], int]:
        """
        Gets observation and action dimensions from environment.

        Args:
            env: Gym environment
            config: Configuration object

        Returns:
            Tuple[int, int]: Observation dimension, Action dimension
        """
        try:
            obs_dim = env.observation_space.shape
            action_dim = (
                env.action_space.shape[0]
                if self.config.continuous_action_space
                else env.action_space.n
            )
        except AttributeError as e:
            raise ValueError(f"Invalid environment space configuration: {e}")
        return obs_dim, action_dim

    def env_setup(self) -> tuple[gym.Env, tuple[int, ...], int]:
        env = self.create_env()
        obs_dim, action_dim = self.get_env_dims(env)
        return env, obs_dim, action_dim
