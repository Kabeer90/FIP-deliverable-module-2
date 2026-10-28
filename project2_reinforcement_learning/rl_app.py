"""
DQN Physics Optimizer -- Streamlit Application
===============================================
Loads a trained DQN agent (dqn_agent.pth), runs one episode of CartPole-v1,
captures every rendered frame, and displays the result as an animated GIF.

Run with:
    streamlit run rl_app.py
"""

import io
import gymnasium as gym
import imageio
import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. DQN ARCHITECTURE (must mirror the training notebook exactly)
# ---------------------------------------------------------------------------
class DQNetwork(nn.Module):
    """
    Deep Q-Network for the CartPole-v1 environment.

    Input:  4-dimensional state vector (cart position, cart velocity,
            pole angle, pole angular velocity).
    Output: Q-values for each of the 2 actions (push left, push right).
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: state embedding
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),

            # Layer 2: hidden representation
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),

            # Layer 3: Q-value output head
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# 2. MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_agent(path: str = "dqn_agent.pth") -> DQNetwork:
    """Load the trained DQN weights from disk."""
    agent = DQNetwork()
    agent.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# 3. EPISODE RUNNER
# ---------------------------------------------------------------------------
def run_episode(agent: DQNetwork, max_steps: int = 500) -> tuple[list[np.ndarray], float, int]:
    """
    Run a single CartPole episode using the trained agent.

    Returns:
        frames  -- list of RGB numpy arrays (one per timestep)
        reward  -- total accumulated reward for the episode
        steps   -- number of timesteps survived
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _ = env.reset()
    frames: list[np.ndarray] = []
    total_reward = 0.0

    for step in range(max_steps):
        # Capture the rendered frame
        frame = env.render()
        frames.append(frame)

        # Select greedy action (no exploration)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            # Capture the final frame
            frames.append(env.render())
            break

    env.close()
    return frames, total_reward, len(frames)


# ---------------------------------------------------------------------------
# 4. GIF CREATION
# ---------------------------------------------------------------------------
def frames_to_gif(frames: list[np.ndarray], fps: int = 30) -> bytes:
    """Encode a list of RGB frames into an in-memory GIF."""
    buffer = io.BytesIO()
    imageio.mimsave(buffer, frames, format="GIF", fps=fps, loop=0)
    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DQN Physics Optimizer", layout="centered")

    st.title("DQN Physics Optimizer")
    st.markdown(
        "This application loads a Deep Q-Network agent trained on the "
        "CartPole-v1 environment.  Click the button below to run one "
        "episode and watch the agent balance the pole."
    )

    agent = load_agent()

    if st.button("Run Episode"):
        with st.spinner("Running episode..."):
            frames, total_reward, steps = run_episode(agent)
            gif_bytes = frames_to_gif(frames, fps=30)

        st.subheader("Episode Results")
        col1, col2 = st.columns(2)
        col1.metric(label="Total Reward", value=f"{total_reward:.0f}")
        col2.metric(label="Steps Survived", value=str(steps))

        st.subheader("Episode Replay")
        st.image(gif_bytes, caption="CartPole-v1 episode rendered by the trained DQN agent.")

        if total_reward >= 475:
            st.success(
                f"The agent achieved a reward of {total_reward:.0f}, which meets "
                "the CartPole-v1 solved threshold (>= 475 over 100 episodes)."
            )
        else:
            st.warning(
                f"The agent achieved a reward of {total_reward:.0f}.  "
                "Individual episode scores may vary; the solved criterion is an "
                "average of >= 475 over 100 consecutive episodes."
            )


if __name__ == "__main__":
    main()
