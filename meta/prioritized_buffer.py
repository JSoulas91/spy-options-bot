import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha  # how much prioritization is used (0 = uniform, 1 = full)
        self.epsilon = epsilon  # small constant to avoid zero priority
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

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= np.max(weights)

        return {
            'states':      [s[0] for s in samples],
            'actions':     [s[1] for s in samples],
            'rewards':     [s[2] for s in samples],
            'next_states': [s[3] for s in samples],
            'dones':       [s[4] for s in samples],
            'weights':     weights,
            'indices':     indices
        }

    def sample_balanced(self, batch_size, beta=0.4, high_rew_cutoff=1.0, low_rew_cutoff=-1.0):
        if not self.buffer:
            raise ValueError("Replay buffer is empty!")

        priorities = np.array(self.priorities, dtype=np.float32)
        rewards = np.array([entry[2] for entry in self.buffer])

        high_idxs = np.where(rewards >= high_rew_cutoff)[0]
        low_idxs = np.where(rewards <= low_rew_cutoff)[0]
        mid_idxs = np.where((rewards < high_rew_cutoff) & (rewards > low_rew_cutoff))[0]

        def sample_idxs(indices, count):
            if len(indices) == 0:
                return []
            local_priorities = priorities[indices]
            probs = local_priorities / local_priorities.sum()
            return np.random.choice(indices, size=min(count, len(indices)), replace=False, p=probs).tolist()

        third = batch_size // 3
        high = sample_idxs(high_idxs, third)
        mid = sample_idxs(mid_idxs, third)
        low = sample_idxs(low_idxs, batch_size - len(high) - len(mid))  # fill remainder

        final_idxs = high + mid + low
        if len(final_idxs) < batch_size:
            print(f"⚠️ Only sampled {len(final_idxs)} out of {batch_size} requested due to limited data")

        np.random.shuffle(final_idxs)

        selected = [self.buffer[i] for i in final_idxs]
        sampled_priorities = priorities[final_idxs]
        probs = sampled_priorities / np.sum(sampled_priorities)

        weights = (len(self.buffer) * probs) ** (-beta)
        weights /= np.max(weights)

        return {
            'states':      [s[0] for s in selected],
            'actions':     [s[1] for s in selected],
            'rewards':     [s[2] for s in selected],
            'next_states': [s[3] for s in selected],
            'dones':       [s[4] for s in selected],
            'weights':     weights,
            'indices':     final_idxs
        }

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            scalar_error = abs(float(error))  # safely convert tensor/float to scalar
            priority = (scalar_error + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)