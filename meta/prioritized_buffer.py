import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        scaled_priorities = priorities ** self.alpha
        sample_probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        samples = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        beta = self.beta_by_frame()
        weights = (len(self.buffer) * sample_probs[indices]) ** (-beta)
        weights /= weights.max()

        self.frame += 1

        return (
            list(states),


Here is the full corrected **`meta/prioritized_buffer.py`** with proper TD error indexing and priority updating logic:

---

### ✅ `meta/prioritized_buffer.py` — Full Replacement

```python
import random
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    """
    Simple Prioritized Experience Replay (PER) buffer using proportional prioritization.
    Stores tuples of (state, action_dir, target_conf, reward, done, next_state, log_prob).
    """

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, transition, priority=1.0):
        self.buffer.append(transition)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("PER buffer is empty!")

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** -self.beta
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        """
        Update sample priorities after training step.
        """
        for i, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[i] = priority

    def __len__(self):
        return len(self.buffer)