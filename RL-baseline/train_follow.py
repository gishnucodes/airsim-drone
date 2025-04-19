import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv # If not using make_vec_env helper

# Import your custom environment
from airsim_follow_env import AirSimFollowEnv # Assuming airsim_follow_env.py is in the same directory

# --- Configuration ---
LOG_DIR = "./training_logs/"
SAVE_PATH = "./trained_follower_model.zip"
TOTAL_TIMESTEPS = 100000 # You will likely need many more timesteps (millions) for good performance

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)

# --- Create the AirSim Environment ---
# Use make_vec_env to create a vectorized environment (even for a single instance)
# This is often more compatible with SB3 algorithms
env = make_vec_env(AirSimFollowEnv, n_envs=1)
# Or without make_vec_env:
# env = AirSimFollowEnv()
# env = DummyVecEnv([lambda: env]) # Wrap if needed by algorithm/callback


# --- Define the RL Model ---
# PPO (Proximal Policy Optimization) is a good general-purpose algorithm
# 'MlpPolicy' means a simple Multi-Layer Perceptron (neural network) policy
# verbose=1 prints training progress
# tensorboard_log=LOG_DIR saves logs for TensorBoard visualization
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR)


# --- Train the Model ---
print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
start_time = time.time()

model.learn(total_timesteps=TOTAL_TIMESTEPS)

end_time = time.time()
print(f"Training finished. Duration: {end_time - start_time:.2f} seconds")

# --- Save the Trained Model ---
model.save(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# --- Close the Environment ---
# The make_vec_env automatically handles closing environments
env.close()
# If using DummyVecEnv directly:
# env.close()

print("Training script finished.")