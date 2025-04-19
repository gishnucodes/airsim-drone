import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import time
import math

class AirSimFollowEnv(gym.Env):
    """
    An AirSim environment for training a drone (Drone2) to follow another drone (Drone1).
    """
    def __init__(self, ip_address="127.0.0.1",
                 drone1_name="Drone1", drone2_name="Drone2",
                 image_shape=(84, 84, 1),  # Example if using vision - not used in this state space
                 max_episode_steps=300,    # Max steps per episode
                 target_altitude=-5,       # Target altitude for flight (negative Z in NED)
                 max_speed=10,             # Maximum speed for Drone2
                 min_follow_distance=2,    # Ideal minimum follow distance for reward
                 max_follow_distance=15,   # Max distance before significant penalty
                 bounds_min_x=-100, bounds_max_x=100, # Simple flight boundaries
                 bounds_min_y=-100, bounds_max_y=100):

        super().__init__()

        self.client = airsim.MultirotorClient(ip=ip_address)
        self.client.confirmConnection()
        print("AirSim client connected to env")

        self.drone1_name = drone1_name
        self.drone2_name = drone2_name
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.target_altitude = target_altitude
        self.max_speed = max_speed
        self.min_follow_distance = min_follow_distance
        self.max_follow_distance = max_follow_distance

        self.bounds_min = np.array([bounds_min_x, bounds_min_y, self.target_altitude - 30]) # Allow some vertical movement
        self.bounds_max = np.array([bounds_max_x, bounds_max_y, self.target_altitude + 10])

        # --- Define State Space (Observation Space) ---
        # Let's use:
        # 1. Relative position vector from Drone2 to Drone1 (x, y, z)
        # 2. Relative velocity vector of Drone1 to Drone2 (vx, vy, vz)
        # 3. Drone2's own linear velocity (vx, vy, vz)
        # Total dimensions: 3 + 3 + 3 = 9

        # Define ranges for normalization/scaling if needed, here using reasonable bounds
        # Relative positions could be large, relative velocities up to 2*max_speed, own velocity up to max_speed
        low_obs = np.array([-200, -200, -100,  # Relative position (x, y, z) - Adjust bounds as needed
                            -self.max_speed*2, -self.max_speed*2, -self.max_speed*2, # Relative velocity
                            -self.max_speed, -self.max_speed, -self.max_speed], dtype=np.float32) # Drone2's velocity
        high_obs = np.array([200, 200, 100,   # Relative position (x, y, z)
                             self.max_speed*2, self.max_speed*2, self.max_speed*2, # Relative velocity
                             self.max_speed, self.max_speed, self.max_speed], dtype=np.float32) # Drone2's velocity

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)


        # --- Define Action Space ---
        # Let's use continuous actions representing desired linear velocities for Drone2
        # Action: [vx, vy, vz]
        self.action_space = spaces.Box(low=np.array([-self.max_speed, -self.max_speed, -self.max_speed], dtype=np.float32),
                                       high=np.array([self.max_speed, self.max_speed, self.max_speed], dtype=np.float32),
                                       dtype=np.float32)

        # State for Drone1's simple random movement
        self._drone1_target_pos = None
        self._drone1_move_speed = 5 # Speed for Drone1's random movement

    def _get_observation(self):
        """Gets the current observation (state) for the agent."""
        # Get state of both drones
        state1 = self.client.getMultirotorState(vehicle_name=self.drone1_name)
        state2 = self.client.getMultirotorState(vehicle_name=self.drone2_name)

        pos1 = state1.kinematics_estimated.position
        vel1 = state1.kinematics_estimated.linear_velocity

        pos2 = state2.kinematics_estimated.position
        vel2 = state2.kinematics_estimated.linear_velocity

        # 1. Relative position (Drone1 - Drone2)
        relative_pos = np.array([pos1.x_val - pos2.x_val,
                                   pos1.y_val - pos2.y_val,
                                   pos1.z_val - pos2.z_val], dtype=np.float32)

        # 2. Relative velocity (Drone1 - Drone2)
        relative_vel = np.array([vel1.x_val - vel2.x_val,
                                   vel1.y_val - vel2.y_val,
                                   vel1.z_val - vel2.z_val], dtype=np.float32)

        # 3. Drone2's own velocity
        drone2_vel = np.array([vel2.x_val, vel2.y_val, vel2.z_val], dtype=np.float32)


        # Combine into a single observation vector
        observation = np.concatenate([relative_pos, relative_vel, drone2_vel])

        return observation

    def _calculate_reward(self, drone1_pos, drone2_pos, action):
        """Calculates the reward based on the current state and action."""
        distance = np.linalg.norm(np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val]) -
                                  np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val]))

        reward = 0

        # Reward for being within the ideal follow distance
        if distance < self.min_follow_distance:
             # Penalize getting too close
             reward += - (self.min_follow_distance - distance) * 0.5 # Small penalty for getting closer than min_distance
        elif distance < self.max_follow_distance:
             # Reward for being within the desired range, peaking at min_follow_distance
             # Example: simple linear reward function
             reward += (self.max_follow_distance - distance) / (self.max_follow_distance - self.min_follow_distance) * 1.0 # Reward up to 1

        else:
            # Significant penalty for being too far
            reward += - (distance - self.max_follow_distance) * 0.2 # Larger penalty the further away it is

        # Small penalty for high action magnitude (encourages efficient movement)
        # action_magnitude = np.linalg.norm(action)
        # reward -= action_magnitude * 0.01 # Adjust weight

        # Check for collisions (simplified: check state)
        state2 = self.client.getMultirotorState(vehicle_name=self.drone2_name)
        if state2.collision.has_collided:
             reward = -100 # Large penalty for collision

        # Check if out of bounds (penalize or terminate)
        # Current Drone2 position
        pos2_np = np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val])
        if np.any(pos2_np < self.bounds_min) or np.any(pos2_np > self.bounds_max):
             reward = -500 # Penalty for going out of bounds
             # Note: You might want to terminate the episode here depending on your goal


        return reward


    def _is_terminated(self, drone2_pos):
        """Checks if the episode is terminated (e.g., crash, out of bounds leading to termination)."""
        state2 = self.client.getMultirotorState(vehicle_name=self.drone2_name)

        # Check for collision
        if state2.collision.has_collided:
            print("Episode terminated due to collision!")
            return True

        # Check if out of bounds (terminate)
        pos2_np = np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val])
        if np.any(pos2_np < self.bounds_min) or np.any(pos2_np > self.bounds_max):
            print("Episode terminated: Drone2 out of bounds!")
            return True

        return False

    def _is_truncated(self):
         """Checks if the episode is truncated (e.g., time limit)."""
         if self.current_step >= self.max_episode_steps:
              print("Episode truncated due to time limit!")
              return True
         return False


    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed) # Important for Gymnasium

        self.current_step = 0

        # Reset the AirSim simulation state
        self.client.reset()
        time.sleep(0.5) # Give simulation a moment to reset

        # Enable API control and arm both drones
        self.client.enableApiControl(True, vehicle_name=self.drone1_name)
        self.client.enableApiControl(True, vehicle_name=self.drone2_name)
        self.client.armDisarm(True, vehicle_name=self.drone1_name)
        self.client.armDisarm(True, vehicle_name=self.drone2_name)
        time.sleep(1) # Wait for arming

        # Takeoff both drones to target altitude
        print(f"Env Reset: Taking off {self.drone1_name} and {self.drone2_name}...")
        
        takeoff_future1 = self.client.takeoffAsync(vehicle_name=self.drone1_name)
        takeoff_future2 = self.client.takeoffAsync(vehicle_name=self.drone2_name)

        takeoff_future1.join()
        takeoff_future2.join()
        print("Env Reset: Takeoff complete.")
        time.sleep(2) # Hover

        # --- Set initial positions if desired (optional, depends on settings.json) ---
        # If you want random start positions within bounds:
        # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(rand_x1, rand_y1, self.target_altitude)), True, vehicle_name=self.drone1_name)
        # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(rand_x2, rand_y2, self.target_altitude)), True, vehicle_name=self.drone2_name)
        # time.sleep(1)


        # --- Initialize Drone1's random movement target ---
        self._set_new_drone1_target()


        # Get initial observation
        observation = self._get_observation()



        info = {} # Optional dictionary for debugging info

        print(f"DEBUG: AirSimFollowEnv.reset preparing to return: observation_type={type(observation)}, info_type={type(info)}")

        return observation, info


    def step(self, action):
        """Performs one step in the environment."""
        self.current_step += 1

        # Ensure action is within bounds and apply to Drone2 (velocity control)
        vx, vy, vz = np.clip(action, -self.max_speed, self.max_speed)
        # Command Drone2's velocity. Use DrivetrainType.ForwardOnly to ignore pitch/roll/yaw for velocity control.
        # The duration is set to a small value (e.g., 0.1s) so we can send new commands frequently in the next step.
        self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), duration=0.1, drivetrain=airsim.DrivetrainType.ForwardOnly, vehicle_name=self.drone2_name)

        # --- Update Drone1's random movement ---
        state1 = self.client.getMultirotorState(vehicle_name=self.drone1_name)
        pos1 = state1.kinematics_estimated.position
        drone1_current_pos_np = np.array([pos1.x_val, pos1.y_val, pos1.z_val])

        # Check if Drone1 reached its random target
        target_pos1_np = np.array([self._drone1_target_pos.x_val, self._drone1_target_pos.y_val, self._drone1_target_pos.z_val])
        distance_to_target1 = np.linalg.norm(drone1_current_pos_np - target_pos1_np)

        if distance_to_target1 < 2.0: # If within 2 meters of target, set a new target
             self._set_new_drone1_target()
             print(f"Drone1 reached target. New target: {self._drone1_target_pos}")
             # Command Drone1 to move to the new target
             self.client.moveToPositionAsync(
                  self._drone1_target_pos.x_val, self._drone1_target_pos.y_val, self._drone1_target_pos.z_val,
                  velocity=self._drone1_move_speed, vehicle_name=self.drone1_name
             )


        # Wait a tiny bit to allow physics to update
        # The duration in moveByVelocityAsync also acts as a form of time step
        # time.sleep(0.05) # You might need this depending on how often you want steps vs velocity command duration


        # Get new observation
        observation = self._get_observation()

        # Get current positions for reward calculation
        state1 = self.client.getMultirotorState(vehicle_name=self.drone1_name)
        state2 = self.client.getMultirotorState(vehicle_name=self.drone2_name)
        pos1 = state1.kinematics_estimated.position
        pos2 = state2.kinematics_estimated.position

        # Calculate reward
        reward = self._calculate_reward(pos1, pos2, action)

        # Check termination and truncation conditions
        terminated = self._is_terminated(pos2)
        truncated = self._is_truncated()

        info = {} # Optional info dict

        return observation, reward, terminated, truncated, info

    def _set_new_drone1_target(self):
        """Sets a new random target position for Drone1 within defined bounds."""
        # Generate a random point within bounds at the target altitude
        rand_x = np.random.uniform(self.bounds_min[0], self.bounds_max[0])
        rand_y = np.random.uniform(self.bounds_min[1], self.bounds_max[1])
        # Keep Z fixed at target altitude for simplicity in this example
        rand_z = self.target_altitude

        self._drone1_target_pos = airsim.Vector3r(rand_x, rand_y, rand_z)

    def close(self):
        """Clean up resources."""
        print("Environment closing. Disarming drones.")
        # Disable API control and disarm on close
        try:
            self.client.enableApiControl(False, vehicle_name=self.drone1_name)
            self.client.enableApiControl(False, vehicle_name=self.drone2_name)
            self.client.armDisarm(False, vehicle_name=self.drone1_name)
            self.client.armDisarm(False, vehicle_name=self.drone2_name)
        except Exception as e:
             print(f"Error during close: {e}")

        # No explicit client.disconnect() in airsim library, usually just let it go out of scope

# --- Helper function to make VecEnv ---
# Stable-Baselines3 typically works better with vectorized environments
# This helper function creates a DummyVecEnv for single environment
def make_airsim_env():
    env = AirSimFollowEnv()
    return env # Simple env for direct use or wrap with DummyVecEnv

# Example of how to wrap with DummyVecEnv (often needed for SB3 algorithms)
# from stable_baselines3.common.env_util import make_vec_env
# vec_env = make_vec_env(AirSimFollowEnv, n_envs=1) # Use this in training script