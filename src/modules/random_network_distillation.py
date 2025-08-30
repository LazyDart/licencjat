from copy import deepcopy

import torch
from tensordict import TensorDict
from torch import nn


class RandomNetworkDistillation(nn.Module):
    """
    Random Network Distillation (RND) for curiosity / intrinsic reward.

    Design goals:
    - Mirror the ICM API where it makes sense so it can be swapped easily.
    - `forward(td)` returns a 3-tuple akin to ICM:
        (inverse_model_logits_or_None, predictor_features, target_features)
      For RND, the first element is None.
    - `calculate_intrinsic_reward(pred, target, eta)` matches ICM’s signature/behavior.
    - Includes `forward_dynamic_model` and `forward_dynamics_loss` for closer parity with ICM.
    - Uses a (frozen) random target network and a trainable predictor network.
    - Optionally detaches the shared encoder output before both nets (stabilizes training).

    Notes:
    - By default, RND uses the observation at t+1 (i.e., `td["next_state"]`) if present,
      otherwise it falls back to `td["state"]`. This is the common choice when using
      RND as an intrinsic reward for the *next* state’s novelty.
    """

    def __init__(
        self,
        head: nn.Module,
        feature_dim: int | None = None,
        hidden: int | None = None,
        predictor_network_override: nn.Sequential | None = None,
        target_network_override: nn.Sequential | None = None,
    ):
        """
        Args:
            head: state encoder (same spirit as ICM.head). Output size should be `feature_dim`.
            feature_dim: dimensionality of the encoded state produced by `head`.
            hidden: hidden size for simple MLPs if overrides are not provided.
            predictor_network_override: optional custom predictor network mapping
                feature_dim -> hidden -> feature_dim (or equivalent).
            target_network_override: optional custom target network of the same I/O shape as predictor.
            detach_encoder: if True, detach encoder output before feeding predictor/target to
                stabilize (prevents predictor from chasing a moving target if head is trained elsewhere).
        """
        super().__init__()

        self.head: nn.Module = head

        if (predictor_network_override is None or target_network_override is None) and (
            feature_dim is None or hidden is None
        ):
            raise ValueError(
                "Provide either (feature_dim and hidden) or both predictor/target overrides."
            )

        if predictor_network_override is not None and target_network_override is not None:
            self.predictor = predictor_network_override
            self.target = target_network_override
        else:
            # Simple MLPs mapping feature_dim -> hidden -> feature_dim
            self.predictor = nn.Sequential(
                self.head,
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, feature_dim),
            )
            self.target = nn.Sequential(
                deepcopy(self.head),
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, feature_dim),
            )

        # Freeze target network (RND's "random" fixed function)
        for p in self.target.parameters():
            p.requires_grad = False

        self._mse = nn.MSELoss()

    def forward_dynamics_loss(
        self, predicted_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """RND prediction loss (MSE between predictor and target)."""
        return self._mse(predicted_features, target_features)

    # ---- Core RND methods ----
    def forward(self, x: TensorDict) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        For swap-ability with ICM, returns a 3-tuple:
            (None, predictor_features, target_features)

        Uses x["next_state"] if available, else x["state"].
        """
        obs = x.get("next_state", None)
        if obs is None:
            obs = x["state"]

        pred = self.predictor(obs)
        tgt = self.target(obs)  # Weights already frozen

        return None, pred, tgt

    def calculate_intrinsic_reward(
        self, pred_features: torch.Tensor, target_features: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """
        Per-sample intrinsic reward from prediction error.
        Matches ICM's signature & scaling (eta * 0.5 * sum of squared error).
        Returns a tensor of shape [B].
        """
        with torch.no_grad():
            return eta * 0.5 * ((pred_features - target_features) ** 2).sum(dim=1)


def rnd_training_step(
    rnd: RandomNetworkDistillation,
    optimizer: torch.optim.Optimizer,
    td: TensorDict,
    eta: float = 0.01,
    device: str | torch.device = "cuda:0",
) -> dict[str, torch.Tensor]:
    """
    One step of RND training, mirroring the style of `icm_training_step`.

    Returns:
        {
          "rnd_loss": scalar float,
          "forward_loss": scalar float (same as rnd_loss),
          "inverse_loss": 0.0 (for logging parity),
          "intrinsic_reward": Tensor[B]
        }
    """
    optimizer.zero_grad()

    rnd = rnd.to(device)
    td = td.to(device)

    _, pred_feat, tgt_feat = rnd(td)

    fwd_loss = rnd.forward_dynamics_loss(pred_feat, tgt_feat)
    fwd_loss.backward()
    optimizer.step()

    intrinsic_reward = rnd.calculate_intrinsic_reward(pred_feat, tgt_feat, eta)

    return {
        "rnd_loss": fwd_loss.item(),
        "forward_loss": fwd_loss.item(),
        "inverse_loss": torch.tensor(0.0, device=device),  # for logging parity with ICM
        "intrinsic_reward": intrinsic_reward,  # tensor [B]
    }
