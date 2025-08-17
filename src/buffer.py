class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    # TODO: perhaps modify the code so that state pairs: prev/next states are kept.
    def store_transition(self, state, action, logprob, reward, done, state_value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.dones.clear()


class RolloutBufferNextState(RolloutBuffer):
    def __init__(self):
        super().__init__()
        self.next_states = []

    def store_transition(self, state, next_state, action, logprob, reward, done, state_value):
        super().store_transition(state, action, logprob, reward, done, state_value)
        self.next_states.append(next_state)

    def clear(self):
        super().clear()
        self.next_states.clear()
