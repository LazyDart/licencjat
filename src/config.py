from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Config(BaseModel):
    # --- training hparams (all required) ---
    num_train_steps: int
    update_interval: int
    num_epochs: int
    batch_size: int
    gamma: float
    eps_clip: float
    action_std_init: float
    action_std_decay_rate: float
    min_action_std: float
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    mode: Literal["train", "eval", "inference"]

    # --- network hparams (all required except ICM group) ---
    agent: Literal["ppo", "ppo_icm"]
    hidden_dim: int
    activation: Literal["tanh", "relu", "gelu", "elu", "leaky_relu"]
    lr_actor: float
    lr_critic: float

    # --- icm_specific_params (only required when agent == "ppo_icm") ---
    lr_icm: float | None = None
    icm_beta: float | None = None
    icm_eta: float | None = None
    intrinsic_coeff: float | None = None

    # --- env hparams (all required) ---
    env_name: str
    continuous_action_space: bool
    max_eps_steps: int
    random_seed: int

    # --- logging hparams ---
    wandb_project: str
    wandb_entity: str
    exp_name: str | None = None
    log_interval: int
    save_interval: int
    log_dir: Path
    ckpt_dir: Path

    # --- evaluation hparams ---
    num_eval_eps: int
    eval_ckpt_name: str
    render_mode: str | None = None

    # Pydantic model configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    # --------- Validators ---------
    @field_validator(
        "num_train_steps",
        "update_interval",
        "num_epochs",
        "batch_size",
        "hidden_dim",
        "max_eps_steps",
        "num_eval_eps",
        "log_interval",
        "save_interval",
        "random_seed",
    )
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("gamma")
    @classmethod
    def _gamma_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        return v

    @field_validator("min_action_std", "action_std_init")
    @classmethod
    def _std_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("std values must be > 0")
        return v

    @field_validator("icm_beta", "icm_eta")
    @classmethod
    def _icm_in_0_1(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("must be in [0, 1]")
        return v

    @field_validator("log_dir", "ckpt_dir")
    @classmethod
    def _ensure_path(cls, v: Path) -> Path:
        # Normalize to Path (doesn't create dirs automatically)
        return Path(v)

    # --------- Cross-field logic ---------
    @model_validator(mode="after")
    def _require_icm_when_agent_is_icm(self) -> Config:
        if self.agent == "ppo_icm":
            missing = [
                name
                for name in ("lr_icm", "icm_beta", "icm_eta", "intrinsic_coeff")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"When agent='ppo_icm', these fields must be provided: {', '.join(missing)}"
                )
        return self

    # --------- YAML helpers ---------
    @classmethod
    def load_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Accept both strings and Paths for dirs
        if "log_dir" in data:
            data["log_dir"] = Path(data["log_dir"])
        if "ckpt_dir" in data:
            data["ckpt_dir"] = Path(data["ckpt_dir"])
        return cls.model_validate(data)

    def save_yaml(self, path: str | Path) -> None:
        def _to_serializable(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            return obj

        data = _to_serializable(self.model_dump())
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    # --------- Update helper (supports dotted keys) ---------
    def update(self, updates: dict[str, Any]) -> Config:
        """
        Update fields in-place. Supports dotted keys (e.g., "logging.log_interval")
        for future nested structures; with the current flat schema, normal keys are used.
        Values are type-validated thanks to `validate_assignment=True`.
        """
        for key, value in updates.items():
            if "." in key:
                key = key.split(".")[-1]
            if not hasattr(self, key):
                raise AttributeError(f"Unknown config key: {key}")
            setattr(self, key, value)
        # Trigger cross-field checks
        self._require_icm_when_agent_is_icm()
        return self


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Load from YAML
    cfg = Config.load_yaml("config.yaml")

    # Update a few things (dotted works too)
    cfg.update(
        {
            "mode": "train",
            "log_dir": "./logs/",
            "save_interval": 200_000,
            "training.batch_size": 2048,  # dotted is accepted; maps to 'batch_size' in this flat schema
        }
    )

    # Save back to YAML
    cfg.save_yaml("config_out.yaml")

    # Access like any Pydantic model
    print(cfg.lr_actor, cfg.env_name, cfg.ckpt_dir)
