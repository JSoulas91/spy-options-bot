# meta/prioritized_buffer.py

import random
import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, state, action, reward, next_state, done, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], [], []

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            list(states),
            list(actions),
            list(rewards),
            list(next_states),
            list(dones),
            weights,
            indices
        )

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()