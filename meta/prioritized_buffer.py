import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, done, error=1.0):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size, beta=None):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        if beta is None:
            beta = self.beta

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)