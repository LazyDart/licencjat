import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module implementation in discrete action space.
    """

    def __init__(
        self,
        head: nn.Module,
        action_space: int,
        feature_dim: int | None = None,
        hidden: int | None = None,
        inverse_model_network_override: nn.Sequential | None = None,
        next_state_pred_network_override: nn.Sequential | None = None,
    ):
        super().__init__()

        self.action_space: int = action_space  # corresponds to output_size
        self.head: nn.Module = head  # Some kind of nn.Module that returns encoded state.

        if feature_dim is not None and hidden is not None:
            self.inverse_model_network = nn.Sequential(
                nn.Linear(2 * feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_space),
            )
            self.next_state_pred_network = nn.Sequential(
                nn.Linear(feature_dim + action_space, hidden),
                nn.ReLU(),
                nn.Linear(hidden, feature_dim),
            )

        elif (
            inverse_model_network_override is not None
            and next_state_pred_network_override is not None
        ):
            self.inverse_model_network = inverse_model_network_override
            self.next_state_pred_network = next_state_pred_network_override

        else:
            raise ValueError("Either dimensions must be provided or matching neural networks.")

        self.inv_loss_cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def encode(self, x):
        # Basic encoding of the state_t
        return self.head(x)

    def forward_dynamic_model(self, encoded_state, action):
        action_onehot = F.one_hot(action, self.action_space)
        state_action_pair = torch.cat((encoded_state, action_onehot), dim=1)
        return self.next_state_pred_network(state_action_pair)

    def forward_dynamics_loss(self, predicted_next_state, encoded_next_state):
        return F.mse_loss(predicted_next_state, encoded_next_state)

    def inverse_model(self, enc_state, enc_next_state):
        """Should return action logits"""

        concatenated_states = torch.cat((enc_state, enc_next_state), dim=1)
        logits = self.inverse_model_network(concatenated_states)
        return logits

    def inverse_loss(self, logits, action_indx):
        return self.inv_loss_cross_entropy(logits, action_indx)

    def inverse_action_probabilties(self, logits):
        return F.softmax(logits, dim=-1)

    def forward(self, x: TensorDict):
        action = x["action"]
        encoded_state = self.encode(x["state"])
        encoded_next_state = self.encode(x["next_state"])

        inverse_model_logits = self.inverse_model(encoded_state, encoded_next_state)

        pred_next_state = self.forward_dynamic_model(encoded_state, action)

        return inverse_model_logits, pred_next_state, encoded_next_state

    def calculate_intrinsic_reward(self, pred_next_enc, next_state_enc, eta):
        with torch.no_grad():
            return eta * 0.5 * ((pred_next_enc - next_state_enc) ** 2).sum(dim=1)


def icm_training_step(
    icm: ICM,
    optimizer: torch.optim.Optimizer,
    td: TensorDict,
    beta: float = 0.2,
    eta: float = 0.01,
    device="cuda:0",
):
    optimizer.zero_grad()

    icm = icm.to(device)
    td = td.to(device)

    inv_logits, pred_next_enc, next_state_enc = icm(td)

    inv_loss = icm.inverse_loss(inv_logits, td["action"])
    fwd_loss = icm.forward_dynamics_loss(pred_next_enc, next_state_enc)
    loss = (1 - beta) * inv_loss + beta * fwd_loss

    loss.backward()
    optimizer.step()

    intrinsic_reward = icm.calculate_intrinsic_reward(pred_next_enc, next_state_enc, eta)

    return {
        "icm_loss": loss.item(),
        "inverse_loss": inv_loss.item(),
        "forward_loss": fwd_loss.item(),
        "intrinsic_reward": intrinsic_reward,  # tensor [B]
    }
