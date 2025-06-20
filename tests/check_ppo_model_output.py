import torch
from meta.ppo import PPOAgent
import numpy as np

STATE_DIM = 83       # Must match meta_state size
NUM_ACTIONS = 3
BATCH_SIZE = 5

def test_model_output_shapes():
    agent = PPOAgent(state_dim=STATE_DIM)
    dummy_input = torch.randn(BATCH_SIZE, STATE_DIM)

    # ✅ FIXED: use agent.net, not agent.model
    direction_logits, confidence, value = agent.net(dummy_input)

    assert direction_logits.shape == (BATCH_SIZE, NUM_ACTIONS), \
        f"🚨 direction_logits has shape {direction_logits.shape}, expected ({BATCH_SIZE}, {NUM_ACTIONS})"
    assert confidence.shape == (BATCH_SIZE, 1), \
        f"🚨 confidence has shape {confidence.shape}, expected ({BATCH_SIZE}, 1)"
    assert value.shape == (BATCH_SIZE, 1), \
        f"🚨 value has shape {value.shape}, expected ({BATCH_SIZE}, 1)"

    print("✅ PPO model outputs have correct shapes.")
    print(f"   direction_logits: {direction_logits.shape}")
    print(f"   confidence:       {confidence.shape}")
    print(f"   value:            {value.shape}")

if __name__ == "__main__":
    test_model_output_shapes()