import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha  # controls how much prioritization is used
        self.epsilon = epsilon  # small constant to ensure non-zero priority
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, error=1.0):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if not self.buffer:
            raise ValueError("Replay buffer is empty!")

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= np.max(weights)  # normalize

        batch = {
            'states':      [s[0] for s in samples],
            'actions':     [s[1] for s in samples],
            'rewards':     [s[2] for s in samples],
            'next_states': [s[3] for s in samples],
            'dones':       [s[4] for s in samples],
            'weights':     weights,
            'indices':     indices
        }
        return batch

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)