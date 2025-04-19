import airsim
import time
import math

# Define the names of your drones
drone1_name = "Drone1"
drone2_name = "Drone2"

# Define a following distance or offset (optional, for slightly behind)
# For simplicity, this script just makes Drone2 target Drone1's exact position
# following_offset_x = -2 # Meters behind Drone1 (in Drone1's local frame - more complex)
# following_offset_y = 0
# following_offset_z = 0

# Define follow speed for Drone2
follow_velocity = 7 # m/s

# Define how long Drone1 will fly its path
flight_duration = 20 # seconds

# Connect to the AirSim simulator
client = airsim.MultirotorClient()

# Confirm connection
try:
    client.confirmConnection()
    print("AirSim connected!")
except Exception as e:
    print(f"Could not connect to AirSim. Is block.exe running? Error: {e}")
    exit()

try:
    # Enable API control for both drones
    client.enableApiControl(True, vehicle_name=drone1_name)
    client.enableApiControl(True, vehicle_name=drone2_name)
    print(f"API control enabled for {drone1_name} and {drone2_name}")

    # Arm both drones
    client.armDisarm(True, vehicle_name=drone1_name)
    client.armDisarm(True, vehicle_name=drone2_name)
    print(f"Armed {drone1_name} and {drone2_name}")

    # Wait a moment for arming
    time.sleep(1)

    # Takeoff both drones
    print(f"Taking off {drone1_name} and {drone2_name}...")
    takeoff_future1 = client.takeoffAsync(vehicle_name=drone1_name)
    takeoff_future2 = client.takeoffAsync(vehicle_name=drone2_name)

    takeoff_future1.join()
    takeoff_future2.join()
    print("Takeoff complete for both drones.")

    # Hover for a moment
    time.sleep(2)

    # --- Start Drone1's movement path ---
    print(f"Starting flight path for {drone1_name}...")

    # Define path points relative to takeoff position (example: a square)
    # Ensure Z is negative for altitude above ground
    altitude = -5 # meters above start

    path_points = [
        airsim.Vector3r(0, 0, altitude), # Hover point (relative to takeoff)
        airsim.Vector3r(10, 0, altitude), # Move East 10m
        airsim.Vector3r(10, 10, altitude), # Move South 10m (relative to last point)
        airsim.Vector3r(0, 10, altitude),  # Move West 10m (relative to last point)
        airsim.Vector3r(0, 0, altitude)   # Return to start (relative to last point)
    ]

    # Get initial position of Drone1 after takeoff to calculate relative moves correctly
    # For simplicity here, we'll just use absolute positions relative to the origin
    # If using relative moves, you'd need to track current position and add the delta
    # Let's use absolute moves relative to the AirSim origin (Player Start)
    # assuming initial X,Y are 0,0 in settings.json
    start_x1 = 0
    start_y1 = 0
    start_z1 = -2 # Initial altitude from settings.json example

    # Define target points in absolute NED coordinates (relative to Player Start)
    absolute_path_points = [
         airsim.Vector3r(start_x1, start_y1, altitude),       # Point A (near start)
         airsim.Vector3r(start_x1 + 10, start_y1, altitude),  # Point B
         airsim.Vector3r(start_x1 + 10, start_y1 + 10, altitude), # Point C
         airsim.Vector3r(start_x1, start_y1 + 10, altitude),  # Point D
         airsim.Vector3r(start_x1, start_y1, altitude)       # Back to Point A
    ]


    # Send Drone1 on its first segment
    print(f"{drone1_name} moving to first point: {absolute_path_points[0]}")
    move1_future = client.moveToPositionAsync(
        absolute_path_points[0].x_val, absolute_path_points[0].y_val, absolute_path_points[0].z_val,
        velocity=5, vehicle_name=drone1_name
    )
    move1_future.join() # Wait for Drone1 to reach the first point


    # --- Main loop for following ---
    print(f"Starting {drone2_name} following loop for {flight_duration} seconds...")
    start_time = time.time()
    path_index = 0
    next_target_point = absolute_path_points[path_index]


    while time.time() - start_time < flight_duration:
        # Get the current state (and position) of Drone1
        drone1_state = client.getMultirotorState(vehicle_name=drone1_name)
        drone1_pos = drone1_state.kinematics_estimated.position

        # The target for Drone2 is Drone1's current position
        target_pos2 = airsim.Vector3r(drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val)
        # print(f"Drone1 Pos: ({drone1_pos.x_val:.2f}, {drone1_pos.y_val:.2f}, {drone1_pos.z_val:.2f}) -> Drone2 Target: ({target_pos2.x_val:.2f}, {target_pos2.y_val:.2f}, {target_pos2.z_val:.2f})")


        # Command Drone2 to move towards Drone1's position
        # Use moveByPositionAsync - does NOT wait for completion in this loop
        # This keeps sending updates to make Drone2 follow
        client.moveToPositionAsync(
            target_pos2.x_val, target_pos2.y_val, target_pos2.z_val,
            velocity=follow_velocity, vehicle_name=drone2_name
        )

        # Check if Drone1 is close to its current target point and needs a new one
        distance_to_target = math.sqrt(
            (drone1_pos.x_val - next_target_point.x_val)**2 +
            (drone1_pos.y_val - next_target_point.y_val)**2 +
            (drone1_pos.z_val - next_target_point.z_val)**2
        )

        if distance_to_target < 1.0 and path_index < len(absolute_path_points) -1: # If within 1 meter of target
             path_index += 1
             next_target_point = absolute_path_points[path_index]
             print(f"{drone1_name} moving to next point: {next_target_point}")
             # Send Drone1 to the next point (async, don't wait)
             client.moveToPositionAsync(
                 next_target_point.x_val, next_target_point.y_val, next_target_point.z_val,
                 velocity=5, vehicle_name=drone1_name
             )


        # Small delay to control update rate and avoid flooding the sim
        time.sleep(0.1) # Adjust this value (e.g., 0.05 to 0.2) for smoother or faster following


    print("Following loop finished.")

    # --- Land both drones ---
    print(f"Landing {drone1_name} and {drone2_name}...")
    # First, stop any active moves
    client.cancelLastTask(vehicle_name=drone1_name)
    client.cancelLastTask(vehicle_name=drone2_name)
    time.sleep(0.5) # Give cancellation a moment

    land_future1 = client.landAsync(vehicle_name=drone1_name)
    land_future2 = client.landAsync(vehicle_name=drone2_name)

    land_future1.join()
    land_future2.join()
    print("Landing complete for both drones.")

    # Disarm both drones (optional)
    client.armDisarm(False, vehicle_name=drone1_name)
    client.armDisarm(False, vehicle_name=drone2_name)
    print(f"Disarmed {drone1_name} and {drone2_name}")


except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always disable API control when done
    try:
        client.enableApiControl(False, vehicle_name=drone1_name)
        client.enableApiControl(False, vehicle_name=drone2_name)
        print("API control disabled for both drones.")
    except Exception as e:
         print(f"Error disabling API control: {e}")