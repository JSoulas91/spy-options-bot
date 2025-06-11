import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, state, action, reward, next_state, done, error=1.0):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty!")

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            new_priority = (abs(err) + 1e-5) ** self.alpha
            if idx < len(self.priorities):
                self.priorities[idx] = new_priority

    def __len__(self):
        return len(self.buffer)