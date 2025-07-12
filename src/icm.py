import torch
import torchrl

import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):

    def __init__(self):
        super().__init__()
        self.action_space: int = ...
        self.head: nn.Module = ... # Some kind of nn.Module that returns encoded state.

        self.linear_inverse: nn.Linear = ... # (batch, 2*state, action_space)
        self.action_dist_prob_dist: nn.Softmax = ...
        self.sparse_softmax_cross_entropy: ... = ... # (cross entropy between logits and choosen action) line 273 in model.py
        self.inv_loss_cross_entropy: nn.CrossEntropyLoss = nn.CrossEntropyLoss()


    def forward(self, x):
        ...
    
    def encode(self, x):
        # Basic encoding of the state_t
        return self.head(x)
    
    def inverse_model(self, enc_state, enc_next_state):
        concatenated_states = torch.concat((enc_state, enc_next_state))
        logits = self.linear_inverse(concatenated_states)

        return logits
    
    def action_prob_dist(self, logits):
        return self.action_dist_prob_dist(logits)
    
    def inverse_loss(self, logits, action_indx):
        return self.inv_loss_cross_entropy(logits, action_indx) 