import airsim
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
import math
import os

# --- Hyperparameters ---
IMG_HEIGHT = 64
IMG_WIDTH = 84
IMG_CHANNELS = 1 # Use grayscale for simplicity, could be 3 for RGB
ACTION_DIM = 4   # Vx, Vy, Vz, Yaw Rate
CLIP_PARAM = 0.2
GAMMA = 0.99
LAMBDA = 0.95     # GAE lambda
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
PPO_EPOCHS = 10
ENTROPY_COEFF = 0.01
MAX_TIMESTEPS_PER_EPISODE = 500
TARGET_UPDATE_FREQ = 2048 # Number of steps before updating the agent
SAVE_MODEL_FREQ = 50      # Save model every N episodes
LOG_FREQ = 1              # Log rewards every N episodes

# --- Target Drone Behavior (Simple Example) ---
# In a real scenario, Drone2 might have its own complex path or AI.
# Here, let's make it hover or move slightly.
def move_target_drone(client, vehicle_name="Drone2"):
    # Example: Make it move in a small circle or just hover
    # For simplicity, we'll just ensure it's flying.
    # In a real test, you might load a recorded path or implement simple movements.
    try:
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        if not state.landed_state == airsim.LandedState.Landed:
             # Optional: Add slight movement here if needed
             # client.moveByVelocityAsync(vx=0.1, vy=0.1, vz=0, duration=1, vehicle_name=vehicle_name)
             pass
        else:
             print(f"Target {vehicle_name} landed unexpectedly.")
    except Exception as e:
        print(f"Error controlling target drone {vehicle_name}: {e}")


# --- AirSim Environment Wrapper ---
class AirSimEnv:
    def __init__(self, chaser_name="Drone1", target_name="Drone2"):
        self.chaser_name = chaser_name
        self.target_name = target_name
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # Don't enable API control here, do it in reset()

        self.max_velocity = 5.0 # Max speed m/s
        self.max_yaw_rate = 90.0 # Max yaw rate deg/s

        self.target_distance_threshold = 3.0 # Ideal distance to target
        self.collision_penalty = -200
        self.goal_reward = 100
        self.step_penalty = -0.1
        self.distance_reward_scale = 10.0

        self.timesteps = 0


    def reset(self):
        print("Resetting environment...")
        self.client.reset()
        # It might take a moment for the drones to be ready after reset
        time.sleep(1.0)

        # Enable API control and take off
        self.client.enableApiControl(True, self.chaser_name)
        self.client.enableApiControl(True, self.target_name)
        self.client.armDisarm(True, self.chaser_name)
        self.client.armDisarm(True, self.target_name)

        print("Taking off...")
        # Use join() to wait for takeoff completion
        self.client.takeoffAsync(vehicle_name=self.chaser_name).join()
        self.client.takeoffAsync(vehicle_name=self.target_name).join()
        # Move to initial hover positions slightly above ground if needed
        self.client.moveToZAsync(-5, 2, vehicle_name=self.chaser_name).join()
        self.client.moveToZAsync(-5, 2, vehicle_name=self.target_name).join()
        time.sleep(1.0) # Allow settling

        self.timesteps = 0
        print("Reset complete.")
        return self._get_observation()

    def _get_observation(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.chaser_name)

        response = responses[0]

        if response is None or response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
             print("Warning: Failed to get valid image. Returning zeros.")
             # Return a blank image of the correct size
             if IMG_CHANNELS == 1:
                return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
             else:
                return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)


        # Get numpy array
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

        # Reshape, resize and handle channels
        try:
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # Resize
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))

            if IMG_CHANNELS == 1:
                # Convert to grayscale
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                # Add channel dimension
                img_processed = np.expand_dims(img_gray, axis=-1)
            else:
                 img_processed = img_resized # Keep RGB

            # Normalize to [0, 1]
            img_final = img_processed.astype(np.float32) / 255.0
            return img_final

        except Exception as e:
             print(f"Error processing image: {e}")
             print(f"Image shape: {img1d.shape}, Response height/width: {response.height}/{response.width}")
             # Return a blank image on error
             if IMG_CHANNELS == 1:
                return np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
             else:
                return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)


    def step(self, action):
        self.timesteps += 1

        # --- Interpret Action ---
        # Action: [vx, vy, vz, yaw_rate] (normalized between -1 and 1 from numpy array)
        # CONVERT NumPy types to standard Python floats explicitly
        vx = float(action[0] * self.max_velocity)
        vy = float(action[1] * self.max_velocity)
        vz = float(action[2] * self.max_velocity) # Z is up/down in AirSim
        yaw_rate = float(action[3] * self.max_yaw_rate)

        # --- Execute Action ---
        duration = 0.1 # Small duration, relies on frequent steps

        # Optional: Add a try-except block for robustness against simulation errors
        try:
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration=duration, # Arguments are now standard floats
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate), # yaw_rate is also float
                vehicle_name=self.chaser_name
            )

            # Optional: Move the target drone slightly
            move_target_drone(self.client, self.target_name)

            # Allow some time for action to take effect before getting state
            # This sleep duration is crucial and might need tuning based on simulation performance
            # If too short, the drone might not have moved much before the next observation.
            # If too long, the control loop becomes slow.
            time.sleep(duration * 0.5) # Adjust timing as needed

            # --- Get New State ---
            next_state_img = self._get_observation()
            chaser_state = self.client.getMultirotorState(vehicle_name=self.chaser_name)
            target_state = self.client.getMultirotorState(vehicle_name=self.target_name)

            # --- Calculate Reward ---
            reward = self._compute_reward(chaser_state, target_state)

            # --- Check Done ---
            done, reason = self._is_done(chaser_state, target_state, reward)

            if done and reason != "Max steps reached": # Avoid redundant print if max steps is the reason
                 print(f"Episode finished: {reason}")
                 # Consider if landing/disarming is needed here, or if reset handles it sufficiently.
                 # Usually, letting reset handle the state transition is cleaner.

            return next_state_img, reward, done

        except Exception as e:
             print(f"!! Error during AirSim step execution (moveByVelocityAsync or subsequent calls): {e}")
             print(f"!! Attempted Action Velocities: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
             # Handle the error gracefully: return a default state, a penalty, and signal episode end
             blank_obs = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
             # Penalize heavily to discourage states leading to errors
             error_penalty = -150.0
             return blank_obs, error_penalty, True

    def _compute_reward(self, chaser_state, target_state):
        reward = 0.0

        # Basic penalty per step to encourage efficiency
        reward += self.step_penalty

        # Check if states are valid
        if not chaser_state or not target_state or not chaser_state.kinematics_estimated or not target_state.kinematics_estimated:
            print("Warning: Invalid state received for reward calculation.")
            return reward # Return basic step penalty if states are invalid

        chaser_pos = chaser_state.kinematics_estimated.position
        target_pos = target_state.kinematics_estimated.position

        # Distance calculation
        distance = math.sqrt(
            (chaser_pos.x_val - target_pos.x_val)**2 +
            (chaser_pos.y_val - target_pos.y_val)**2 +
            (chaser_pos.z_val - target_pos.z_val)**2
        )

        # Reward for being close (inverse distance, capped)
        dist_reward = self.distance_reward_scale / max(distance, 0.1)
        reward += dist_reward

        # Bonus for being within the target distance threshold
        if distance < self.target_distance_threshold:
            reward += self.goal_reward * (1.0 - distance / self.target_distance_threshold) # More reward closer

        # Collision check (using chaser drone's collision info)
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.chaser_name)
        if collision_info.has_collided:
            reward += self.collision_penalty

        return reward


    def _is_done(self, chaser_state, target_state, reward):
        # Check for collision
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.chaser_name)
        if collision_info.has_collided:
            print(f"Collision detected with object: {collision_info.object_name}")
            return True, "Collision"

        # Check if chaser went out of bounds or got lost (requires defining bounds)
        # Example: Check altitude or distance from origin
        chaser_pos = chaser_state.kinematics_estimated.position
        if abs(chaser_pos.z_val) > 50 or math.sqrt(chaser_pos.x_val**2 + chaser_pos.y_val**2) > 100:
            return True, "Out of bounds"

        # Check if target is too far (chaser lost target)
        target_pos = target_state.kinematics_estimated.position
        distance = math.sqrt(
            (chaser_pos.x_val - target_pos.x_val)**2 +
            (chaser_pos.y_val - target_pos.y_val)**2 +
            (chaser_pos.z_val - target_pos.z_val)**2
        )
        if distance > 50: # Adjust this 'lost' threshold
             return True, "Target lost"

        # Check max timesteps
        if self.timesteps >= MAX_TIMESTEPS_PER_EPISODE:
            return True, "Max steps reached"

        return False, ""

    def close(self):
        print("Landing and Disabling API control...")
        try:
             self.client.landAsync(vehicle_name=self.chaser_name).join()
             self.client.landAsync(vehicle_name=self.target_name).join()
             self.client.armDisarm(False, self.chaser_name)
             self.client.armDisarm(False, self.target_name)
             self.client.enableApiControl(False, self.chaser_name)
             self.client.enableApiControl(False, self.target_name)
        except Exception as e:
             print(f"Error during cleanup: {e}")
        self.client.reset() # Final reset
        print("Environment closed.")


# --- PPO Agent ---
# Simple Buffer class
class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def sample(self, batch_size):
        n_states = len(self.states)
        if n_states == 0:
            return None

        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return (
            np.array(self.states)[indices],
            np.array(self.actions)[indices],
            np.array(self.rewards)[indices],
            np.array(self.dones)[indices],
            np.array(self.log_probs)[indices],
            np.array(self.values)[indices],
            batches
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

# Actor-Critic Network using CNN for visual input
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCritic, self).__init__()

        # CNN Base
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size dynamically
        def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
            from math import floor
            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            if type(stride) is not tuple:
                stride = (stride, stride)
            if type(pad) is not tuple:
                pad = (pad, pad)
            h = floor(((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
            w = floor(((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
            return h, w

        h, w = IMG_HEIGHT, IMG_WIDTH
        h, w = conv_output_size((h, w), kernel_size=8, stride=4)
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        h, w = conv_output_size((h, w), kernel_size=3, stride=1)
        self.flattened_size = h * w * 64

        # Actor Head
        self.actor_fc = nn.Linear(self.flattened_size, 256)
        self.actor_mu = nn.Linear(256, action_dim) # Mean of action distribution
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim)) # Log std deviation

        # Critic Head
        self.critic_fc = nn.Linear(self.flattened_size, 256)
        self.critic_value = nn.Linear(256, 1) # State value estimate

    def forward(self, x):
        # Input x shape: (batch, channels, height, width)
        if len(x.shape) == 3: # Add batch dim if missing
             x = x.unsqueeze(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # --- FIX IS HERE ---
        # Replace .view() with .reshape() for flattening
        # x = x.view(x.size(0), -1) # Original Line
        x = x.reshape(x.size(0), -1) # Corrected Line
        # -------------------

        # Actor Path
        actor_latent = F.relu(self.actor_fc(x))
        action_mean = torch.tanh(self.actor_mu(actor_latent)) # Tanh to constrain mean to [-1, 1]
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Critic Path
        critic_latent = F.relu(self.critic_fc(x))
        state_value = self.critic_value(critic_latent)

        return action_mean, action_std, state_value

# PPO Agent Logic
class PPOAgent:
    def __init__(self, input_channels, action_dim, device):
        self.device = device
        self.policy = ActorCritic(input_channels, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.buffer = Buffer()

    def select_action(self, state):
        # State shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, action_std, state_value = self.policy(state_tensor)

        # Create action distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1) # Sum log probs for multi-dim action

        # Clip action to [-1, 1] range
        action = torch.clamp(action, -1.0, 1.0)

        return action.cpu().numpy().flatten(), log_prob.cpu().item(), state_value.cpu().item()


    def update(self):
        if len(self.buffer.states) < BATCH_SIZE:
            print(f"Skipping update: Buffer size {len(self.buffer.states)} < Batch size {BATCH_SIZE}")
            return

        print("Updating agent...")
        data = self.buffer.sample(BATCH_SIZE)
        if data is None:
            print("No data sampled from buffer.")
            return

        states_np, actions_np, rewards_np, dones_np, old_log_probs_np, values_np, batches = data

        # Calculate GAE (Generalized Advantage Estimation)
        advantages = np.zeros(len(rewards_np), dtype=np.float32)
        last_gae_lam = 0
        # Convert dones to float for calculation
        dones_float = dones_np.astype(float)

        # Ensure values_np has the correct shape and includes the value of the *next* state implicitly
        # For the last state, V(s_T) = 0 if done, else it should be estimated (but often approximated as 0 or handled by bootstrap)
        num_samples = len(rewards_np)
        for t in reversed(range(num_samples)):
            if t == num_samples - 1:
                next_non_terminal = 1.0 - dones_float[t]
                next_values = 0 # Simplified: Assume V=0 for the state after the last stored one
            else:
                next_non_terminal = 1.0 - dones_float[t+1]
                next_values = values_np[t+1]

            delta = rewards_np[t] + GAMMA * next_values * next_non_terminal - values_np[t]
            advantages[t] = last_gae_lam = delta + GAMMA * LAMBDA * next_non_terminal * last_gae_lam

        # Normalize advantages (optional but often helpful)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Calculate returns (target for value function)
        returns = advantages + values_np

        # Convert to tensors
        states = torch.FloatTensor(states_np).permute(0, 3, 1, 2).to(self.device) # (N, C, H, W)
        actions = torch.FloatTensor(actions_np).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs_np).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)


        # Optimize policy for PPO_EPOCHS
        for _ in range(PPO_EPOCHS):
             for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate current policy
                mean, std, values = self.policy(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().mean() # Average entropy across batch and action dims

                # Calculate ratio and surrogate losses
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss - ENTROPY_COEFF * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # Gradient clipping
                self.optimizer.step()

        # Clear buffer after update
        self.buffer.clear()
        print("Agent update complete.")

    def save_model(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: Model file not found at {filepath}. Starting fresh.")


# --- Training Loop ---
def train():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment and agent
    env = AirSimEnv()
    agent = PPOAgent(IMG_CHANNELS, ACTION_DIM, device)

    # Load existing model if available
    model_dir = "airsim_ppo_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_drone_chaser.pth")
    agent.load_model(model_path)


    total_steps = 0
    episode = 0
    try:
        while True: # Run indefinitely or set a max episode/step count
            episode += 1
            state = env.reset()
            episode_reward = 0
            done = False
            steps_in_episode = 0

            while not done and steps_in_episode < MAX_TIMESTEPS_PER_EPISODE:
                # Select action
                action, log_prob, value = agent.select_action(state)

                # Step environment
                next_state, reward, done = env.step(action)

                # Store experience
                agent.buffer.store(state, action, reward, done, log_prob, value)

                # Update state and rewards
                state = next_state
                episode_reward += reward
                total_steps += 1
                steps_in_episode += 1

                # Update agent policy if enough data collected
                if total_steps % TARGET_UPDATE_FREQ == 0 and len(agent.buffer.states) > 0:
                     # Need the value of the *final* next_state for GAE calculation before clearing
                     next_state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
                     with torch.no_grad():
                           _, _, final_value = agent.policy(next_state_tensor)
                     # Append this final value estimate to the buffer's values temporarily for GAE calc?
                     # PPO update uses the values stored during collection, GAE handles the final step.
                     agent.update() # Update handles buffer clearing internally

            # Logging
            if episode % LOG_FREQ == 0:
                 print(f"Episode: {episode}, Steps: {steps_in_episode}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

            # Save model periodically
            if episode % SAVE_MODEL_FREQ == 0:
                agent.save_model(model_path)

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Ensure environment is cleaned up properly
        env.close()
        # Save final model
        agent.save_model(model_path)
        print("Final model saved.")


if __name__ == "__main__":
    # Make sure AirSim is running and the Blocks environment is loaded
    # with the correct settings.json before starting the script.
    print("Starting Drone Chasing Training...")
    print("Ensure AirSim Simulation is running with Blocks environment and correct settings.json.")
    # Add a small delay to ensure user sees the message
    time.sleep(3)
    train()