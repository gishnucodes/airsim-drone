import time
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv # Not used in this test
# from stable_baselines3.common.env_util import make_vec_env # Not used in this test

# Import your custom environment
from airsim_follow_env import AirSimFollowEnv

# --- Configuration ---
MODEL_PATH = "./trained_follower_model.zip"
TEST_EPISODES = 1 # Just run one episode for this test
MAX_TEST_STEPS_PER_EPISODE = 50 # Limit steps for this test

# --- Create the AirSim Environment DIRECTLY for testing ---
# Remove or comment out the make_vec_env line
# env = make_vec_env(AirSimFollowEnv, n_envs=1, env_kwargs={'max_episode_steps': MAX_TEST_STEPS_PER_EPISODE})

# Create the environment instance directly
print("Creating AirSimFollowEnv directly...")
env = AirSimFollowEnv(max_episode_steps=MAX_TEST_STEPS_PER_EPISODE)
print("AirSimFollowEnv created directly.")


# --- Load the Trained Model (optional for this specific reset test, but keep for context) ---
# You might not even reach this if the reset fails
try:
    # model = PPO.load(MODEL_PATH) # Keep commented out for the reset test
    print(f"Model loading skipped for direct reset test")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
    # exit() # Don't exit, continue with reset test


# --- Run Test Episodes (Modified for direct reset test) ---
print(f"Running direct environment reset test for {TEST_EPISODES} episode...")

for episode in range(TEST_EPISODES):
    print(f"\n--- Starting Direct Reset Test Episode {episode + 1} ---")
    try:
        # Call reset directly and try to unpack
        print("Calling env.reset() directly...")
        # This is the line that previously failed when env was a VecEnv
        obs, info = env.reset()
        print(f"SUCCESS: env.reset() returned observation type {type(obs)} and info type {type(info)}")

        # You could technically run steps here, but the goal is just to test reset
        # print("Starting dummy steps...")
        # for step in range(10): # Run a few dummy steps
        #     # Provide a dummy action (e.g., zeros or random) as no model is loaded
        #     dummy_action = env.action_space.sample() # Or np.zeros_like(env.action_space.low)
        #     obs, reward, terminated, truncated, info = env.step(dummy_action)
        #     print(f"Step {step}: Reward={reward}, Terminated={terminated}, Truncated={truncated}")
        #     if terminated or truncated:
        #         break


    except ValueError as e:
        print(f"ERROR: Direct env.reset() failed with ValueError: {e}")
        print("Problem confirmed in the base environment's reset return.")
    except Exception as e:
        print(f"ERROR: Direct env.reset() failed with unexpected error: {e}")
    finally:
        # Ensure close is called
        # env.close() # Call manually if not using wrappers

        print("Direct environment reset test finished.")
        env.close() # Manually close the environment instance