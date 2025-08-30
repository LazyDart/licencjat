import argparse
import os
import time
import warnings
from collections import deque
from typing import TypedDict

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import wandb
from tensordict import TensorDict

from src import utils
from src.agent_factory import AgentFactory
from src.config import Config
from src.env_factory import EnvFactory
from src.ppo import PPOAgentICM, PPOAgentRND

np.random.seed(seed=42)


class MetricsDict(TypedDict):
    eps_rewards: list[float]
    eps_lengths: list[int]
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_eps_length: float
    total_steps: int
    eval_dt: float


def run_training(config: Config, device: str) -> None:
    start_time = time.time()

    env_factory = EnvFactory(config)
    env, obs_dim, action_dim = env_factory.env_setup()
    logger.info(f"Observation Dimension: {obs_dim} | Action Dimension: {action_dim}")

    agent_factory = AgentFactory(obs_dim, action_dim, config, device)
    ppo_agent = agent_factory.make_agent()

    running_eps_reward: float = 0
    running_eps_intrinsic_reward: float = 0
    running_eps_length: float = 0
    running_num_eps: float = 0

    # start training loop
    t_so_far: int = 0
    eps_so_far: int = 0

    while t_so_far < config.num_train_steps:
        obs, _ = env.reset(seed=config.random_seed)

        obs = torch.from_numpy(obs)
        eps_reward: float = 0
        eps_intrinsic_reward: float = 0
        eps_length: int = 0

        # start episode
        for _ in range(1, config.max_eps_steps + 1):
            # print("Observation:", obs.shape)
            action, logprob, value = ppo_agent.policy.select_action(obs)
            # print("Action:", action.shape, action.dtype, "Logprob:", logprob.shape, logprob.dtype, "Value:", value)
            next_obs, reward, done, _, info = env.step(action)
            next_obs = torch.from_numpy(next_obs)

            eps_reward += reward
            t_so_far += 1
            eps_length += 1

            # store transitions in buffer
            if config.agent == "ppo":
                ppo_agent.buffer.store_transition(obs, action, logprob, reward, done, value)
            elif isinstance(ppo_agent, PPOAgentICM) or isinstance(ppo_agent, PPOAgentRND):
                ppo_agent.buffer.store_transition(
                    obs, action, logprob, reward, done, value, next_state=next_obs
                )
            else:
                raise NotImplementedError("Other agent not implemented")

            if t_so_far % config.update_interval == 0:
                if isinstance(ppo_agent, PPOAgentICM) or isinstance(ppo_agent, PPOAgentRND):
                    td = TensorDict(
                        {
                            "action": torch.tensor(ppo_agent.buffer.actions).long().to(device),
                            "state": torch.tensor(ppo_agent.buffer.states).to(device),
                            "next_state": torch.tensor(ppo_agent.buffer.next_states).to(device),
                        }
                    )
                    with torch.no_grad():
                        _, pred_next_enc, next_state_enc = ppo_agent.int_rew_model(td)
                        intrinsic_reward = (
                            ppo_agent.int_rew_model.calculate_intrinsic_reward(
                                pred_next_enc, next_state_enc, ppo_agent.int_rew_eta
                            )
                            .detach()
                            .cpu()
                        ) * config.intrinsic_coeff  # [T]
                        ppo_agent.buffer.rewards[: intrinsic_reward.shape[0]] += intrinsic_reward
                        eps_intrinsic_reward += float(intrinsic_reward.sum().item())
                else:
                    intrinsic_reward = 0

                if config.freeze_threshold is not None and t_so_far < config.freeze_threshold:
                    ppo_agent.stop_updates = True

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
                if isinstance(ppo_agent, PPOAgentICM) or isinstance(ppo_agent, PPOAgentRND):
                    checkpoint_path_icm = os.path.join(
                        config.ckpt_dir, f"{config.env_name}_intrinsic_model_step_{t_so_far}.pt"
                    )
                    torch.save(ppo_agent.int_rew_model.state_dict(), checkpoint_path_icm)

            obs = next_obs
            if done:
                break

        running_eps_reward += eps_reward
        running_eps_intrinsic_reward += eps_intrinsic_reward
        running_eps_length += eps_length
        running_num_eps += 1
        eps_so_far += 1

    env.close()
    wandb.finish()  # close wandb logging
    print(f"Training time: {(time.time() - start_time) / 60.0:.2f} mins")


import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


def run_evaluation(
    config: Config, device: str, save_video: bool = True, verbose: bool = True
) -> MetricsDict:
    """
    Evaluates a trained agent on the given environment and (optionally) saves video(s).
    """

    metrics: MetricsDict = {
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

    # --- Build base env to get dims (your factory) ---
    env_factory = EnvFactory(config)
    env, obs_dim, action_dim = env_factory.env_setup()
    logger.info(f"Observation Dimension: {obs_dim} | Action Dimension: {action_dim}")

    # --- Swap to a video-capable env if requested ---
    # MiniGrid/BabyAI need render_mode picked at construction time for RGB frames.
    if save_video:
        # Close the factory env and rebuild with rgb_array for recording
        env.close()
        env = gym.make(config.env_name, render_mode="rgb_array")
        import minigrid

        env = minigrid.wrappers.ImgObsWrapper(env)
        # Where to save MP4s
        video_dir = os.path.join(
            "/home/lazydart/Python Codes/licencjat/logs/BabyAI-FindObjS7-v0/runs/run_20250823T131905/checkpoints/",
            "eval_videos",
        )
        os.makedirs(video_dir, exist_ok=True)

        # Record every episode
        env = RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix="eval",
        )
        print(env)
        logger.info(f"Recording evaluation videos to: {video_dir}")

    # --- Agent setup ---
    agent_factory = AgentFactory(obs_dim, action_dim, config, device)
    ppo_agent = agent_factory.make_agent()

    # --- Load model ---
    ckpt_path = "/home/lazydart/Python Codes/licencjat/logs/BabyAI-FindObjS7-v0/runs/run_20250823T131905/checkpoints/BabyAI-FindObjS7-v0_policy_step_400000.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ppo_agent.policy.load_state_dict(ckpt)
        if isinstance(ppo_agent, PPOAgentICM):
            ckpt_icm_path = "/home/lazydart/Python Codes/licencjat/logs/BabyAI-FindObjS7-v0/runs/run_20250823T131905/checkpoints/BabyAI-FindObjS7-v0_icm_step_400000.pt"
            ckpt = torch.load(ckpt_icm_path, map_location=device)
            ppo_agent.int_rew_model.load_state_dict(ckpt)
        logger.info(f"Successfully loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")

    ppo_agent.policy.eval()
    recent_rewards: deque = deque(maxlen=10)
    start_time = time.time()

    with torch.no_grad():
        for eps_so_far in range(config.num_eval_eps):
            # Gymnasium reset: obs, info
            obs, _ = env.reset()
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs)
            eps_reward = 0.0
            eps_length = 0

            for step in range(config.max_eps_steps):
                # Policy step
                action, _, _ = ppo_agent.policy.select_action(obs)

                # Gymnasium step returns (obs, reward, terminated, truncated, info)
                step_out = env.step(action)
                next_obs, reward, done, _, info = step_out

                if not isinstance(next_obs, torch.Tensor):
                    next_obs = torch.from_numpy(next_obs)

                eps_reward += float(reward)
                eps_length += 1
                obs = next_obs

                if done:
                    break

            # Update metrics
            metrics["eps_rewards"].append(eps_reward)
            metrics["eps_lengths"].append(eps_length)
            metrics["min_reward"] = min(metrics["min_reward"], eps_reward)
            metrics["max_reward"] = max(metrics["max_reward"], eps_reward)
            metrics["total_steps"] += eps_length
            recent_rewards.append(eps_reward)

            # Early stopping if rewards converge
            if len(recent_rewards) == recent_rewards.maxlen:
                if np.std(recent_rewards) < 0.1 * max(1e-8, np.mean(recent_rewards)):
                    logger.info("Early stopping as rewards have converged")
                    break

    metrics["eval_dt"] = time.time() - start_time
    metrics["mean_reward"] = (
        float(np.mean(metrics["eps_rewards"])) if metrics["eps_rewards"] else 0.0
    )
    metrics["std_reward"] = float(np.std(metrics["eps_rewards"])) if metrics["eps_rewards"] else 0.0
    metrics["mean_eps_length"] = (
        float(np.mean(metrics["eps_lengths"])) if metrics["eps_lengths"] else 0.0
    )

    if verbose:
        logger.info("\nEvaluation Summary:")
        logger.info(f"Mean Reward: {metrics['mean_reward']:.2f} +- {metrics['std_reward']:.2f}")
        logger.info(f"Min/Max Reward: {metrics['min_reward']:.2f}/{metrics['max_reward']:.2f}")
        logger.info(f"Mean Episode Length: {metrics['mean_eps_length']:.2f}")
        logger.info(f"Total Steps: {metrics['total_steps']}")
        logger.info(f"Eval Time: {metrics['eval_dt']:.2f} s")

        if save_video:
            logger.info("Videos saved under the 'eval_videos' folder (MP4 files per episode).")

    env.close()
    return metrics


def parse_args() -> argparse.Namespace:
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


def main(config: Config) -> None:
    device = utils.set_device()
    logger.info(f"using device: {device}")
    # logger.info(config)

    # create environment
    logger.info(f"Environment: {config.env_name}")

    if config.random_seed:
        utils.set_random_seed(config.random_seed)

    run_id = "run_" + time.strftime("%Y%m%dT%H%M%S")
    config.log_dir = config.log_dir / config.env_name / "runs" / run_id
    config.ckpt_dir = config.log_dir / "checkpoints"

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

        run_training(config, device)

        # save config file for each run in log directory
        Config.save_config(config, os.path.join(config.log_dir, "config.yaml"))

    elif config.mode == "test":
        run_evaluation(config, device, verbose=args.verbose)

    else:
        logger.error("Invalid mode. Mode should be either 'train' or 'test'.")

    print("Done!")


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
