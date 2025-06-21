import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, sequence_length, state_dim, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def _pad_sequence(self, seq):
        if len(seq) >= self.sequence_length:
            return np.array(seq[-self.sequence_length:])
        else:
            pad_len = self.sequence_length - len(seq)
            pad = [np.zeros(self.state_dim)] * pad_len
            return np.array(pad + seq)

    def add(self, past_states, action, reward, past_next_states, done, error=1.0):
        """
        Adds full sequences of shape (seq_len, state_dim).
        Assumes inputs are either lists or np arrays of shape (seq_len, state_dim).
        """
        state_seq = self._pad_sequence(past_states)
        next_state_seq = self._pad_sequence(past_next_states)
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.buffer.append((state_seq, action, reward, next_state_seq, done))
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
            'states':      np.stack([s[0] for s in samples]),  # (B, T, D)
            'actions':     np.array([s[1] for s in samples]),
            'rewards':     np.array([s[2] for s in samples], dtype=np.float32),
            'next_states': np.stack([s[3] for s in samples]),
            'dones':       np.array([s[4] for s in samples], dtype=np.float32),
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
        low = sample_idxs(low_idxs, batch_size - len(high) - len(mid))

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
            'states':      np.stack([s[0] for s in selected]),
            'actions':     np.array([s[1] for s in selected]),
            'rewards':     np.array([s[2] for s in selected], dtype=np.float32),
            'next_states': np.stack([s[3] for s in selected]),
            'dones':       np.array([s[4] for s in selected], dtype=np.float32),
            'weights':     weights,
            'indices':     final_idxs
        }

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            scalar_error = abs(float(error))
            priority = (scalar_error + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)