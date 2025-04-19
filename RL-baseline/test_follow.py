import time
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv # If needed
from stable_baselines3.common.env_util import make_vec_env

# Import your custom environment
from airsim_follow_env import AirSimFollowEnv

# --- Configuration ---
MODEL_PATH = "./trained_follower_model.zip"
TEST_EPISODES = 5 # Number of test episodes to run
MAX_TEST_STEPS_PER_EPISODE = 500 # Max steps for a test episode

# --- Create the AirSim Environment (should match training setup) ---
# Use make_vec_env even for testing for consistency
# env = make_vec_env(AirSimFollowEnv, n_envs=1, env_kwargs={'max_episode_steps': MAX_TEST_STEPS_PER_EPISODE})
# Or:
env = AirSimFollowEnv(max_episode_steps=MAX_TEST_STEPS_PER_EPISODE)
# env = DummyVecEnv([lambda: env]) # Wrap if needed


# --- Load the Trained Model ---
try:
    model = PPO.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
    exit()


# --- Run Test Episodes ---
print(f"Running {TEST_EPISODES} test episodes...")

for episode in range(TEST_EPISODES):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps_in_episode = 0

    print(f"\n--- Starting Episode {episode + 1} ---")

    while not done and not truncated:
        # Predict action from the trained model
        action, _states = model.predict(obs, deterministic=True) # deterministic=True uses the mean action

        # Step the environment
        obs, reward, done, truncated, info = env.step(action)

        episode_reward += reward
        steps_in_episode += 1

        # Optional: Add a small delay to visualize better
        time.sleep(0.05)

        # You can print state/reward/info here for debugging if needed
        # print(f"Step: {steps_in_episode}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")


    print(f"--- Episode {episode + 1} finished after {steps_in_episode} steps with total reward: {episode_reward:.2f} ---")

# --- Close the Environment ---
env.close()

print("Testing script finished.")