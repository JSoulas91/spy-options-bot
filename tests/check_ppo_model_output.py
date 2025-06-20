import torch
from meta.ppo import PPOAgent  # adjust if your agent is elsewhere
import numpy as np

STATE_DIM = 83       # Must match meta_state size
NUM_ACTIONS = 3      # Assuming short/neutral/long
BATCH_SIZE = 5       # Test with small batch

def test_model_output_shapes():
    # 🔧 Initialize PPO agent
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=NUM_ACTIONS
    )

    # 🧪 Generate dummy input
    dummy_input = torch.randn(BATCH_SIZE, STATE_DIM)

    # 🔁 Forward pass
    direction_logits, confidence, value = agent.model(dummy_input)

    # ✅ Assertions
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