import random
import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, errors):
        for i in range(len(indices)):
            val = indices[i]
            if isinstance(val, (np.ndarray, list)):
                if np.size(val) != 1:
                    raise ValueError(f"Cannot convert index of shape {np.shape(val)} to scalar")
                idx = val.item()
            else:
                idx = int(val)

            err = errors[i]
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)