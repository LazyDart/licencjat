
from typing import Any, Tuple, Optional, Dict
import torch

class RolloutBuffer:
    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.state_values: list[float] = []
        self.dones: list[bool] = []

    # NOTE: kwargs may include next_state in subclasses
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        state_value: float,
        **kwargs: Any,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(float(reward))
        self.state_values.append(float(state_value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()


class RolloutBufferNextState(RolloutBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.next_states: list[torch.Tensor] = []

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        state_value: float,
        **kwargs: Any,
    ) -> None:
        super().store_transition(state, action, logprob, reward, done, state_value)
        # required in this subclass
        self.next_states.append(kwargs["next_state"])  # type: ignore[index]

    def clear(self) -> None:
        super().clear()
        self.next_states.clear()


class TensorRolloutBuffer:
    """
    Preallocated tensor buffer for fixed-horizon rollouts.
    All fields are tensors on CPU (optionally pinned). Dtypes are consistent:
      - states, next_states: float32
      - actions: int64 (discrete) or float32 (continuous)
      - logprobs, rewards, values: float32
      - dones: bool
    """
    def __init__(
        self,
        horizon: int,
        obs_shape: Tuple[int, ...],
        action_shape: Optional[Tuple[int, ...]],
        *,
        discrete_actions: bool,
        store_next_state: bool = False,
        pin_memory: bool = True,
    ) -> None:
        self.H = int(horizon)
        self.discrete = bool(discrete_actions)
        self.store_next = bool(store_next_state)

        self.states      = torch.empty((self.H, *obs_shape), dtype=torch.float32, pin_memory=pin_memory)
        if self.discrete:
            self.actions = torch.empty((self.H,), dtype=torch.int64, pin_memory=pin_memory)
        else:
            assert action_shape is not None and len(action_shape) == 1, "continuous action must be 1D"
            self.actions = torch.empty((self.H, action_shape[0]), dtype=torch.float32, pin_memory=pin_memory)

        self.logprobs    = torch.empty((self.H,), dtype=torch.float32, pin_memory=pin_memory)
        self.rewards     = torch.empty((self.H,), dtype=torch.float32, pin_memory=pin_memory)
        self.state_vals  = torch.empty((self.H,), dtype=torch.float32, pin_memory=pin_memory)
        self.dones       = torch.empty((self.H,), dtype=torch.bool,   pin_memory=pin_memory)

        self.next_states = None
        if self.store_next:
            self.next_states = torch.empty((self.H, *obs_shape), dtype=torch.float32, pin_memory=pin_memory)

        self.idx = 0   # write pointer
        self.full = False

    def store_transition(
        self,
        *,
        state: torch.Tensor,          # [obs...], float32, CPU
        action: torch.Tensor,         # [] int64 (discrete) or [A] float32, CPU
        logprob: torch.Tensor,        # [] float32, CPU
        reward: float,
        done: bool,
        state_value: torch.Tensor,    # [] float32, CPU
        next_state: Optional[torch.Tensor] = None,
    ) -> None:
        i = self.idx
        self.states[i].copy_(state.to(torch.float32))
        if self.discrete:
            # accept scalar tensor ([]) or shaped (), coerce to int64
            self.actions[i] = action.to(torch.int64)
        else:
            self.actions[i].copy_(action.to(torch.float32))

        # accept 0-D tensors or floats
        self.logprobs[i]   = float(logprob)
        self.rewards[i]    = float(reward)
        self.state_vals[i] = float(state_value)
        self.dones[i]      = bool(done)

        if self.store_next:
            assert self.next_states is not None
            if next_state is None:
                raise ValueError("next_state is required when store_next_state=True")
            self.next_states[i].copy_(next_state.to(torch.float32))

        self.idx += 1
        if self.idx >= self.H:
            self.idx = 0
            self.full = True

    def batch(self, n_valid: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Returns a view (no copy) of the first n_valid rows.
        If n_valid is None: use H when full, else current idx.
        """
        n = self.H if (n_valid is None and self.full) else (n_valid or self.idx)
        sl = slice(0, n)
        out = {
            "state": self.states[sl],
            "action": self.actions[sl],
            "logprob": self.logprobs[sl],
            "reward": self.rewards[sl],
            "value": self.state_vals[sl],
            "done": self.dones[sl],
        }
        if self.store_next:
            out["next_state"] = self.next_states[sl]  # type: ignore[index]
        return out

    def clear(self) -> None:
        self.idx = 0
        self.full = False
        # zeroing is not required; idx/n_valid gates validity
        # TODO don't trust this way of doing things. Verify it is consistent with ppo implementation