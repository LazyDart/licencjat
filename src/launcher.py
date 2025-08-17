import argparse
import os
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=DeprecationWarning)

import gymnasium as gym
import minigrid
import numpy as np
import torch
import wandb
from tensordict import TensorDict

from src import utils
from src.config import Config
from src.modules.feature_extractors import MinigridFeaturesExtractor
from src.ppo import PPOAgent, PPOAgentICM


def make_agent(
    obs_dim: int,
    action_dim: int,
    config: Config,
    device: str,
) -> PPOAgent:
    if "minigrid" in config.env_name.lower():
        obs_dim = obs_dim
        feature_extractor = MinigridFeaturesExtractor(obs_dim, config.hidden_dim)
    else:
        feature_extractor = None
        obs_dim = obs_dim[0]

    if config.agent.lower() == "ppo":
        # initialize a PPO agent
        ppo_agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            continuous_action_space=config.continuous_action_space,
            num_epochs=config.num_epochs,
            eps_clip=config.eps_clip,
            action_std_init=config.action_std_init,
            gamma=config.gamma,
            entropy_coef=config.entropy_coef,
            value_loss_coef=config.value_loss_coef,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            device=device,
            feature_extractor_override=feature_extractor,
        )
        return ppo_agent

    elif config.agent.lower() == "ppo_icm":
        ppo_agent = PPOAgentICM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            lr_actor=config.lr_actor,
            lr_critic=config.lr_critic,
            continuous_action_space=config.continuous_action_space,
            num_epochs=config.num_epochs,
            eps_clip=config.eps_clip,
            action_std_init=config.action_std_init,
            gamma=config.gamma,
            entropy_coef=config.entropy_coef,
            value_loss_coef=config.value_loss_coef,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            device=device,
            icm_beta=config.icm_beta,
            icm_eta=config.icm_eta,
            lr_icm=config.lr_icm,
            feature_extractor_override=feature_extractor,
        )
        return ppo_agent
    else:
        raise ValueError(f"Invalid agent name: {config.agent}")


def create_env(env_name: str) -> gym.Env:
    try:
        env = gym.make(env_name)
        if "minigrid" in env_name.lower():
            env = minigrid.wrappers.ImgObsWrapper(env)
    except:
        raise ValueError(f"Invalid environment name: {env_name}")
    return env


def get_env_dims(env, config) -> tuple[tuple[int, ...], int]:
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
            env.action_space.shape[0] if config.continuous_action_space else env.action_space.n
        )
        logger.info(f"Observation Dimension: {obs_dim} | Action Dimension: {action_dim}")
    except AttributeError as e:
        raise ValueError(f"Invalid environment space configuration: {e}")
    return obs_dim, action_dim


def run_training(env: gym.Env, config: Config, device: str):
    start_time = time.time()

    obs_dim, action_dim = get_env_dims(env, config)
    ppo_agent = make_agent(obs_dim, action_dim, config, device)

    # training metrics
    metrics = {
        "eps_rewards": [],
        "eps_lengths": [],
        "mean_reward": 0,
        "mean_eps_length": 0,
        "num_episodes": 0,
    }

    running_eps_reward = 0
    running_eps_intrinsic_reward = 0
    running_eps_length = 0
    running_num_eps = 0

    # start training loop
    t_so_far = 0
    eps_so_far = 0

    while t_so_far < config.num_train_steps:
        obs, _ = env.reset(seed=config.random_seed)
        eps_reward = 0
        eps_intrinsic_reward = 0
        eps_length = 0

        # start episode
        for _ in range(1, config.max_eps_steps + 1):
            # print("Observation:", obs.shape)
            action, logprob, value = ppo_agent.policy.select_action(obs)
            # print("Action:", action.shape, action.dtype, "Logprob:", logprob.shape, logprob.dtype, "Value:", value)
            next_obs, reward, done, _, info = env.step(action)

            eps_reward += reward
            t_so_far += 1
            eps_length += 1

            # store transitions in buffer
            if config.agent == "ppo":
                ppo_agent.buffer.store_transition(obs, action, logprob, reward, done, value)
            elif config.agent == "ppo_icm":
                ppo_agent.buffer.store_transition(
                    obs, next_obs, action, logprob, reward, done, value
                )
            else:
                raise NotImplementedError("Other agent not implemented")

            if t_so_far % config.update_interval == 0:
                if config.agent == "ppo_icm":
                    td = TensorDict(
                        {
                            "action": torch.tensor(np.array(ppo_agent.buffer.actions))
                            .long()
                            .to(device),
                            "state": torch.tensor(np.array(ppo_agent.buffer.states)).to(device),
                            "next_state": torch.tensor(np.array(ppo_agent.buffer.next_states)).to(
                                device
                            ),
                        }
                    )
                    with torch.no_grad():
                        _, pred_next_enc, next_state_enc = ppo_agent.icm(td)
                        intrinsic_reward = (
                            ppo_agent.icm.calculate_intrinsic_reward(
                                pred_next_enc, next_state_enc, ppo_agent.icm_eta
                            )
                            .cpu()
                            .numpy()
                        )
                        for i in range(len(intrinsic_reward)):
                            ppo_agent.buffer.rewards[i] += intrinsic_reward[i]
                            eps_intrinsic_reward += intrinsic_reward[i]
                else:
                    intrinsic_reward = 0

                ppo_agent.update_weights()

            if t_so_far % config.log_interval == 0:
                running_eps_reward /= running_num_eps
                running_eps_intrinsic_reward /= running_num_eps
                running_eps_length /= running_num_eps

                logger.info(
                    f"episode: {eps_so_far} | step: {t_so_far} | reward: {running_eps_reward:.4f} | episode length: {running_eps_length}"
                )

                with open(os.path.join(config.log_dir, "log.txt"), "a") as f:
                    f.write(
                        f"episode: {eps_so_far} | step: {t_so_far} | reward: {running_eps_reward:.4f} | episode length: {running_eps_length}\n"
                    )

                wandb.log(
                    {
                        "mean_episode_reward": running_eps_reward,
                        "mean_episode_intrinsic_reward": running_eps_intrinsic_reward,
                        "mean_episode_length": running_eps_length,
                        "episode": eps_so_far,
                        "total_steps": t_so_far,
                    },
                    step=t_so_far,
                )

                running_eps_reward = 0
                running_eps_intrinsic_reward = 0
                running_eps_length = 0
                running_num_eps = 0

            if t_so_far % config.save_interval == 0:
                checkpoint_path_policy = os.path.join(
                    config.ckpt_dir, f"{config.env_name}_policy_step_{t_so_far}.pt"
                )
                torch.save(ppo_agent.policy.state_dict(), checkpoint_path_policy)
                if config.agent == "ppo_icm":
                    checkpoint_path_icm = os.path.join(
                        config.ckpt_dir, f"{config.env_name}_icm_step_{t_so_far}.pt"
                    )
                    torch.save(ppo_agent.icm.state_dict(), checkpoint_path_icm)

            obs = next_obs
            if done:
                break

        metrics["eps_rewards"].append(eps_reward)
        metrics["eps_lengths"].append(eps_length)

        running_eps_reward += eps_reward
        running_eps_intrinsic_reward += eps_intrinsic_reward
        running_eps_length += eps_length
        running_num_eps += 1
        eps_so_far += 1

    wandb.finish()  # close wandb logging
    print(f"Training time: {(time.time() - start_time) / 60.0:.2f} mins")


def run_evaluation(
    env, config: Config, device: str, save_video: bool = False, verbose: bool = True
):
    """
    Evaluates a trained agent on the given environment
    """

    obs_dim, action_dim = get_env_dims(env, config)

    metrics = {
        "eps_rewards": [],
        "eps_lengths": [],
        "mean_reward": 0,
        "std_reward": 0,
        "min_reward": float("inf"),
        "max_reward": float("-inf"),
        "mean_eps_length": 0,
        "total_steps": 0,
        "eval_dt": 0,
    }

    # create agent
    ppo_agent = make_agent(obs_dim, action_dim, config, device)

    # load model
    ckpt_path = os.path.join(config.ckpt_dir, config.eval_ckpt_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # print(ckpt.keys())
        ppo_agent.policy.load_state_dict(ckpt)
        logger.info(f"Successfully loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

    # set policy to evaluation mode
    ppo_agent.policy.eval()
    recent_rewards = deque(maxlen=10)  # Track recent rewards for early stopping
    start_time = time.time()

    with torch.no_grad():
        for eps_so_far in range(config.num_eval_eps):
            obs, _ = env.reset()
            eps_reward = 0
            eps_length = 0

            # start episode
            for step in range(config.max_eps_steps):
                if config.render_mode:
                    env.render()

                action, logprob, value = ppo_agent.policy.select_action(obs)
                next_obs, reward, done, _, info = env.step(action.item())

                eps_reward += reward
                eps_length += 1

                obs = next_obs
                if done:
                    break

            # update metrics
            metrics["eps_rewards"].append(eps_reward)
            metrics["eps_lengths"].append(eps_length)
            metrics["min_reward"] = min(metrics["min_reward"], eps_reward)
            metrics["max_reward"] = max(metrics["max_reward"], eps_reward)
            metrics["total_steps"] += eps_length
            recent_rewards.append(eps_reward)

            # Early stopping check
            if len(recent_rewards) == recent_rewards.maxlen:
                if np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                    logger.info("Early stopping as rewards have converged")
                    break

    metrics["eval_dt"] = time.time() - start_time
    metrics["mean_reward"] = np.mean(metrics["eps_rewards"])
    metrics["std_reward"] = np.std(metrics["eps_rewards"])
    metrics["mean_eps_length"] = np.mean(metrics["eps_lengths"])

    if verbose:
        logger.info("\nEvaluation Summary:")
        logger.info(f"Mean Reward: {metrics['mean_reward']:.2f} +- {metrics['std_reward']:.2f}")
        logger.info(f"Min/Max Reward: {metrics['min_reward']:.2f}/{metrics['max_reward']:.2f}")
        logger.info(f"Mean Episode Length: {metrics['mean_eps_length']:.2f}")
        logger.info(f"Total Steps: {metrics['total_steps']}")
        logger.info(f"Eval Time: {metrics['eval_dt']:.2f} s")

    env.close()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--config_path", type=str, required=True, help="path to config file")
    parser.add_argument("--wandb_dir", type=str, default=None, help="path to wandb directory")
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")

    # Allow overriding any config parameter from command line
    parser.add_argument(
        "--override", nargs="*", default=[], help="override parameters, format: key=value"
    )
    args = parser.parse_args()
    return args


def main(config):
    device = utils.set_device()
    logger.info(f"using device: {device}")
    # logger.info(config)

    # create environment
    logger.info(f"Environment: {config.env_name}")
    env = create_env(config.env_name)

    if config.random_seed:
        utils.set_random_seed(config.random_seed)

    run_id = "run_" + time.strftime("%Y%m%dT%H%M%S")
    config.log_dir = os.path.join(config.log_dir, config.env_name, "runs", run_id)
    config.ckpt_dir = os.path.join(config.log_dir, "checkpoints")

    logger.info(f"Mode: {config.mode}")
    if config.mode == "train":
        # initialize wandb for logging
        run = wandb.init(
            project=config.wandb_project,
            name=config.exp_name or f"{config.env_name}-{time.strftime('%Y%m%dT%H%M%S')}",
            config=config,  # Track hyperparameters
            dir=args.wandb_dir,
            monitor_gym=False,  # Auto-log gym environment videos
        )
        logger.info("wandb initialized...")

        # create the logs directory if it doesn't exist
        # os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)

        log_path = os.path.join(config.log_dir, "log.txt")
        logger.info(f"Logs saved at: {log_path}")

        run_training(env, config, device)

        # save config file for each run in log directory
        Config.save_config(config, os.path.join(config.log_dir, "config.yaml"))

    elif config.mode == "test":
        run_evaluation(env, config, device, verbose=args.verbose)

    else:
        logger.error("Invalid mode. Mode should be either 'train' or 'test'.")

    print("Done!")
    env.close()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse_args()

    if args.wandb_dir is None:
        args.wandb_dir = os.path.join(os.getcwd(), "../")
    # os.makedirs(args.wandb_dir, exist_ok=True)

    # load configuration
    config = Config.load_yaml(args.config_path)

    # override with command-line arguments
    overrides = {}
    for override in args.override:
        key, value = override.split("=")
        try:
            value = eval(value)
        except:
            pass
        overrides[key] = value
    config = Config.update(config, overrides)

    # Set up logging
    logger = utils.setup_logging(log_dir=config.log_dir, verbose=args.verbose)

    main(config)
