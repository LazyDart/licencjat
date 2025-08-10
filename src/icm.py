import torch
import torchrl
from tensordict import TensorDict

import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module implementation in discrete action space.
    """

    def __init__(self, head: nn.Module, action_space: int):
        super().__init__()
        self.action_space: int = action_space # corresponds to output_size
        self.head: nn.Module = head # Some kind of nn.Module that returns encoded state.

        self.inverse_model_network: nn.Sequential = ... # (batch, 2*state) and return (batch, action_space) logits
        self.sparse_softmax_cross_entropy: ... = ... # (cross entropy between logits and choosen action) line 273 in model.py
        self.inv_loss_cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        self.next_state_pred_network: nn.Sequential = ... # (batch, cat(encode_state, action_space))

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
        action_probabilities = self.inverse_action_probabilties(inverse_model_logits)

        pred_next_state = self.forward_dynamic_model(encoded_state, action)
        
        return inverse_model_logits, pred_next_state, encoded_next_state#, action_probabilities #TODO is it needed?
    
def icm_training_step(icm: ICM, optimizer, td, beta: float = 0.2, eta: float = 0.01):
    optimizer.zero_grad()

    inv_logits, pred_next_enc, next_state_enc = icm(td)

    inv_loss = icm.inverse_loss(inv_logits, td["action"])
    fwd_loss = icm.forward_dynamics_loss(pred_next_enc, next_state_enc)
    loss = ... #

    loss.backward()
    optimizer.step()

    # intrinsic reward often: eta * 0.5 * ||phi_hat - phi||^2 per-sample
    with torch.no_grad():
        intrinsic_reward = eta * 0.5 * ((pred_next_enc - next_state_enc) ** 2).sum(dim=1) # TODO: check it...

    return {
        "loss": loss.item(),
        "inverse_loss": inv_loss.item(),
        "forward_loss": fwd_loss.item(),
        "intrinsic_reward": intrinsic_reward,  # tensor [B]
    }
