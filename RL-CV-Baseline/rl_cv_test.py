import airsim
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# NOTE: No optimizer or torch.distributions needed for testing
import time
import math
import os

# --- Parameters (Should match training settings where applicable) ---
IMG_HEIGHT = 64
IMG_WIDTH = 84
IMG_CHANNELS = 1 # IMPORTANT: Set to 1 for grayscale, 3 for RGB (must match trained model)
ACTION_DIM = 4   # Vx, Vy, Vz, Yaw Rate
MODEL_PATH = "airsim_ppo_models/ppo_drone_chaser.pth" # Path to the trained model weights
NUM_TEST_EPISODES = 10 # How many episodes to run for testing
MAX_STEPS_PER_TEST_EPISODE = 600 # Max steps per episode during testing

# --- Reusable ActorCritic Network Definition ---
# IMPORTANT: This MUST be identical to the ActorCritic class definition used during training
#            including the input_channels and action_dim.
class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(ActorCritic, self).__init__()

        # CNN Base (ensure architecture matches training)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Helper function to calculate output size of convolutions
        def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
            from math import floor
            # Ensure kernel_size, stride, pad are tuples
            if type(kernel_size) is not tuple: kernel_size = (kernel_size, kernel_size)
            if type(stride) is not tuple: stride = (stride, stride)
            if type(pad) is not tuple: pad = (pad, pad)
            # Calculate output height and width
            h = floor(((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
            w = floor(((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
            return h, w

        # Calculate the flattened size dynamically based on convolutions
        h, w = IMG_HEIGHT, IMG_WIDTH
        h, w = conv_output_size((h, w), kernel_size=8, stride=4)
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        h, w = conv_output_size((h, w), kernel_size=3, stride=1)
        self.flattened_size = h * w * 64 # 64 is the number of output channels from conv3

        # Actor Head
        self.actor_fc = nn.Linear(self.flattened_size, 256)
        self.actor_mu = nn.Linear(256, action_dim) # Outputs mean of the action distribution
        # Log standard deviation (parameter, needed for loading state_dict)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic Head (needed for loading state_dict, not used for action selection in test)
        self.critic_fc = nn.Linear(self.flattened_size, 256)
        self.critic_value = nn.Linear(256, 1) # Outputs state value estimate

    # Define the forward pass
    def forward(self, x):
        # Ensure input has batch dimension if missing
        if len(x.shape) == 3: x = x.unsqueeze(0)
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the output for fully connected layers (use reshape for compatibility)
        x = x.reshape(x.size(0), -1)

        # --- Actor Path ---
        actor_latent = F.relu(self.actor_fc(x))
        # Output action mean, constrained by tanh to [-1, 1]
        action_mean = torch.tanh(self.actor_mu(actor_latent))
        # Calculate action standard deviation (not used for deterministic action but needed for model structure)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # --- Critic Path ---
        critic_latent = F.relu(self.critic_fc(x))
        # Output state value (not used for action selection in test)
        state_value = self.critic_value(critic_latent)

        # Return mean for deterministic action selection during testing
        # Also return std and value for potential analysis if needed
        return action_mean, action_std, state_value


# --- Simplified AirSim Environment for Testing ---
class AirSimTestEnv:
    """Handles interaction with AirSim for testing purposes."""
    def __init__(self, chaser_name="Drone1", target_name="Drone2"):
        self.chaser_name = chaser_name
        self.target_name = target_name
        self.client = airsim.MultirotorClient()
        print("Connecting to AirSim...")
        try:
            self.client.confirmConnection()
            print("Connected!")
            # Ensure API control is initially off for a clean reset
            self.client.enableApiControl(False, self.chaser_name)
            self.client.enableApiControl(False, self.target_name)
            self.client.reset()
            print("Initial reset done.")
        except Exception as e:
            print(f"Error connecting to or resetting AirSim: {e}")
            print("Please ensure AirSim is running and accessible.")
            raise # Re-raise exception to stop script if connection fails

        # Drone control parameters (match training)
        self.max_velocity = 5.0
        self.max_yaw_rate = 90.0
        self.timesteps = 0

    def reset(self):
        """Resets the simulation state for a new test episode."""
        print("Resetting environment for test episode...")
        try:
            # Reset the simulation
            self.client.reset()
            time.sleep(1.0) # Give time for reset to settle

            # Enable API control and arm drones
            self.client.enableApiControl(True, self.chaser_name)
            self.client.enableApiControl(True, self.target_name)
            self.client.armDisarm(True, self.chaser_name)
            self.client.armDisarm(True, self.target_name)
            print("API Control Enabled and Drones Armed.")

            # Take off and move to a starting altitude
            print("Taking off...")
            # Use join() to wait for async operations to complete
            takeoff_chaser = self.client.takeoffAsync(vehicle_name=self.chaser_name)
            takeoff_target = self.client.takeoffAsync(vehicle_name=self.target_name)
            takeoff_chaser.join()
            takeoff_target.join()
            print("Takeoff complete. Moving to initial altitude...")
            move_chaser = self.client.moveToZAsync(-5, 2, vehicle_name=self.chaser_name) # Move to Z=-5m
            move_target = self.client.moveToZAsync(-5, 2, vehicle_name=self.target_name) # Move target too
            move_chaser.join()
            move_target.join()
            time.sleep(1.5) # Allow drones to stabilize at altitude
            print("Drones at initial altitude.")

        except Exception as e:
             print(f"Error during environment reset (takeoff/move): {e}")
             print("Attempting to continue, but simulation state might be inconsistent.")
             # Try to ensure API control is enabled if possible after error
             try:
                 if not self.client.isApiControlEnabled(vehicle_name=self.chaser_name):
                     self.client.enableApiControl(True, self.chaser_name)
                 if not self.client.isApiControlEnabled(vehicle_name=self.target_name):
                     self.client.enableApiControl(True, self.target_name)
             except Exception as api_e:
                  print(f"Failed to re-enable API control after reset error: {api_e}")

        self.timesteps = 0
        print("Environment reset complete.")
        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        """Gets and preprocesses the camera image from the chaser drone."""
        try:
            # Request scene image from the front camera ("0")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ], vehicle_name=self.chaser_name)
            response = responses[0]

            # Check if image data is valid
            if response is None or response.image_data_uint8 is None or len(response.image_data_uint8) == 0:
                 print("Warning: Failed to get valid image data from AirSim. Returning zeros.")
                 # Return a blank image matching the expected dimensions
                 shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
                 return np.zeros(shape, dtype=np.float32)

            # Process the image data
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3) # Reshape to HxWx3
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT)) # Resize

            # Handle color channels (grayscale or RGB)
            if IMG_CHANNELS == 1:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                # Add channel dimension: (H, W) -> (H, W, 1)
                img_processed = np.expand_dims(img_gray, axis=-1)
            else:
                 img_processed = img_resized # Keep as (H, W, 3) for RGB

            # Normalize pixel values to [0, 1]
            img_final = img_processed.astype(np.float32) / 255.0
            return img_final

        except Exception as e:
             print(f"Error getting/processing observation: {e}")
             # Return a blank image on error
             shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
             return np.zeros(shape, dtype=np.float32)


    def step(self, action):
        """Executes an action, gets the next state, and checks for basic done conditions."""
        self.timesteps += 1

        # --- Interpret and Scale Action ---
        # Convert normalized action [-1, 1] to velocity/yaw rate
        # Explicitly convert to standard Python float for AirSim API
        vx = float(action[0] * self.max_velocity)
        vy = float(action[1] * self.max_velocity)
        vz = float(action[2] * self.max_velocity) # Note: In AirSim, positive Z is usually down.
        yaw_rate = float(action[3] * self.max_yaw_rate)

        duration = 0.1 # Duration for velocity command
        done = False
        reason = ""

        try:
            # --- Execute Action ---
            self.client.moveByVelocityAsync(
                vx, vy, vz, duration=duration,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, # Use appropriate drivetrain
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate), # Control yaw rate
                vehicle_name=self.chaser_name
            )
            # NOTE: No need to .join() here, let it execute while we wait/get obs

            # Optional: Implement simple movement for the target drone during testing
            # self.move_target_drone_simple()

            # Wait for a short period to allow the action to have an effect
            time.sleep(duration * 0.6) # Adjust timing multiplier as needed

            # --- Get Next Observation ---
            next_state_img = self._get_observation()

            # --- Check Done Conditions (Simplified for Testing) ---
            # 1. Collision Check
            collision_info = self.client.simGetCollisionInfo(vehicle_name=self.chaser_name)
            if collision_info.has_collided:
                print(f"Collision detected! Object ID: {collision_info.object_id}, Name: {collision_info.object_name}")
                done = True
                reason = "Collision"

            # 2. Max Steps Check
            if self.timesteps >= MAX_STEPS_PER_TEST_EPISODE:
                if not done: # Avoid overwriting collision reason
                    done = True
                    reason = "Max steps reached"

            # 3. Optional: Out of Bounds Check (Example)
            # chaser_state = self.client.getMultirotorState(vehicle_name=self.chaser_name)
            # chaser_pos = chaser_state.kinematics_estimated.position
            # if abs(chaser_pos.z_val) > 50 or math.sqrt(chaser_pos.x_val**2 + chaser_pos.y_val**2) > 150:
            #    if not done: # Avoid overwriting previous reasons
            #        done = True
            #        reason = "Out of bounds"

            return next_state_img, done, reason

        except Exception as e:
             print(f"!! Error during AirSim step execution: {e}")
             # Return default values to end the episode gracefully
             shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
             blank_obs = np.zeros(shape, dtype=np.float32)
             return blank_obs, True, "Error in step execution"


    def close(self):
        """Lands drones, disables API control, and resets simulation."""
        print("Test finished. Cleaning up AirSim environment...")
        try:
             # Check if API control is enabled before trying to land/disarm
             if self.client.isApiControlEnabled(vehicle_name=self.chaser_name):
                 print(f"Landing {self.chaser_name}...")
                 self.client.landAsync(vehicle_name=self.chaser_name).join()
                 self.client.armDisarm(False, self.chaser_name)
             if self.client.isApiControlEnabled(vehicle_name=self.target_name):
                 print(f"Landing {self.target_name}...")
                 self.client.landAsync(vehicle_name=self.target_name).join()
                 self.client.armDisarm(False, self.target_name)
        except Exception as e:
             print(f"Error during landing/disarming: {e}")
        finally:
             # Ensure API control is disabled and reset simulation
             try:
                 self.client.enableApiControl(False, self.chaser_name)
                 self.client.enableApiControl(False, self.target_name)
                 self.client.reset()
                 print("API control disabled and simulation reset.")
             except Exception as e:
                 print(f"Error during final cleanup (disable API/reset): {e}")
        print("Environment closed.")

    # Optional helper for simple target movement during test
    # def move_target_drone_simple(self):
    #     try:
    #         # Example: Make target hover or move slightly
    #         self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.target_name)
    #     except Exception as e:
    #         print(f"Error moving target drone: {e}")


# --- Main Testing Function ---
def test_model():
    """Loads the trained model and runs test episodes in AirSim."""
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the testing environment
    try:
        env = AirSimTestEnv()
    except Exception as e:
        # Error during connection/initial reset is handled in __init__
        return # Exit if environment setup failed

    # Initialize the policy network (must match the saved model's architecture)
    policy = ActorCritic(IMG_CHANNELS, ACTION_DIM).to(device)

    # Load the trained weights
    if os.path.exists(MODEL_PATH):
        try:
            # Load state dict, mapping to the correct device
            policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Model weights loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights from {MODEL_PATH}: {e}")
            print("Ensure the model file is valid and matches the network architecture.")
            env.close() # Clean up environment before exiting
            return
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the path is correct and the model file exists.")
        env.close() # Clean up environment before exiting
        return

    # Set the network to evaluation mode
    # This disables dropout layers and uses running averages for batch normalization
    policy.eval()

    # --- Run Test Episodes ---
    try:
        for episode in range(NUM_TEST_EPISODES):
            print(f"\n--- Starting Test Episode {episode + 1}/{NUM_TEST_EPISODES} ---")
            # Reset environment and get initial state
            state = env.reset()
            done = False
            steps = 0

            # Loop within the episode
            while not done:
                steps += 1
                # Preprocess state: Convert to tensor, permute dims (H,W,C) -> (C,H,W), add batch dim
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)

                # Get deterministic action from the policy network
                # No gradient calculation needed during evaluation
                with torch.no_grad():
                    # We only need the action mean for deterministic behavior
                    action_mean, _, _ = policy(state_tensor)

                # Convert action tensor to numpy array and remove batch dimension
                action = action_mean.cpu().numpy().flatten()
                # Clip action to ensure it's within the valid range [-1, 1]
                action = np.clip(action, -1.0, 1.0)

                # Execute action in the environment
                next_state, done, reason = env.step(action)

                # --- Optional: Display Chaser's View ---
                # Uncomment to show the drone's camera feed in a window
                # try:
                #     # Denormalize image from [0,1] to [0,255] and convert to uint8
                #     display_img = (state * 255).astype(np.uint8)
                #     # Convert grayscale to BGR if necessary for cv2.imshow
                #     if IMG_CHANNELS == 1 and len(display_img.shape) == 3 and display_img.shape[2] == 1:
                #        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                #     # Ensure it's a valid image before showing
                #     if display_img is not None and display_img.size > 0:
                #         cv2.imshow("Chaser View (Testing)", display_img)
                #     # Check for 'q' key press to quit (waitKey returns -1 if no key pressed)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #        print("User pressed 'q', stopping test.")
                #        done = True
                #        reason = "User quit"
                # except Exception as display_e:
                #     print(f"Error displaying image: {display_e}")
                # ----------------------------------------

                # Update state for the next iteration
                state = next_state

                # Check if episode finished
                if done:
                    print(f"Episode {episode + 1} finished after {steps} steps. Reason: {reason}")

            # Short pause between episodes
            if episode < NUM_TEST_EPISODES - 1: # Don't pause after the last episode
                 print("Pausing before next episode...")
                 time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing loop: {e}")
    finally:
        # Ensure environment is properly closed
        env.close()
        # Close any OpenCV windows if they were opened
        # cv2.destroyAllWindows()
        print("Testing script finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    print("="*30)
    print("   AirSim Drone Chaser - TEST RUN")
    print("="*30)
    print(f"Attempting to load model: {MODEL_PATH}")
    print(f"Running for {NUM_TEST_EPISODES} episodes.")
    print("\nEnsure AirSim (Blocks environment) is running with the correct settings.json.")
    print("Press Ctrl+C in the console to interrupt testing.")
    # Give user time to read messages and ensure AirSim is ready
    time.sleep(4)
    test_model()
