import airsim
import time
import argparse # Import argparse to handle potential command-line arguments later if needed

# Connect to the AirSim simulator
# Use the ip_address argument if connecting to a remote machine (your remote desktop's IP)
# Otherwise, it defaults to "127.0.0.1" (localhost)
client = airsim.MultirotorClient()

# Confirm connection
try:
    client.confirmConnection()
    print("AirSim connected!")
except Exception as e:
    print(f"Could not connect to AirSim. Is block.exe running? Error: {e}")
    exit()

# --- Define the names of your drones as specified in settings.json ---
drone1_name = "Drone1"
drone2_name = "Drone2"

try:
    # Enable API control for both drones
    client.enableApiControl(True, vehicle_name=drone1_name)
    client.enableApiControl(True, vehicle_name=drone2_name)
    print(f"API control enabled for {drone1_name} and {drone2_name}")

    # Arm both drones
    client.armDisarm(True, vehicle_name=drone1_name)
    client.armDisarm(True, vehicle_name=drone2_name)
    print(f"Armed {drone1_name} and {drone2_name}")

    # Wait a moment for arming to complete
    time.sleep(1)

    # Takeoff both drones
    print(f"Taking off {drone1_name} and {drone2_name}...")
    # takeoffAsync returns a Future object, allowing simultaneous commands
    takeoff_future1 = client.takeoffAsync(vehicle_name=drone1_name)
    takeoff_future2 = client.takeoffAsync(vehicle_name=drone2_name)

    # Wait for both takeoff tasks to complete
    takeoff_future1.join()
    takeoff_future2.join()
    print("Takeoff complete for both drones.")

    # Wait a moment in the air
    time.sleep(3)

    # --- Fly the drones to different positions ---
    print(f"Moving {drone1_name} and {drone2_name} to target positions...")

    # Example movement: Move Drone1 forward (negative X in NED) and Drone2 East (positive Y)
    # The Z value is negative for altitude above the ground (NED: Down is positive Z)
    # The 5 parameter is the velocity (m/s)
    move_future1 = client.moveToPositionAsync(x=-10, y=0, z=-5, velocity=5, vehicle_name=drone1_name)
    move_future2 = client.moveToPositionAsync(x=0, y=10, z=-5, velocity=5, vehicle_name=drone2_name)

    # Wait for both movement tasks to complete
    move_future1.join()
    move_future2.join()
    print("Movement complete for both drones.")

    # Wait at the new positions
    time.sleep(3)

    # --- Land both drones ---
    print(f"Landing {drone1_name} and {drone2_name}...")
    land_future1 = client.landAsync(vehicle_name=drone1_name)
    land_future2 = client.landAsync(vehicle_name=drone2_name)

    # Wait for both landing tasks to complete
    land_future1.join()
    land_future2.join()
    print("Landing complete for both drones.")

    # Disarm both drones (optional)
    client.armDisarm(False, vehicle_name=drone1_name)
    client.armDisarm(False, vehicle_name=drone2_name)
    print(f"Disarmed {drone1_name} and {drone2_name}")

except Exception as e:
    print(f"An error occurred during the flight: {e}")

finally:
    # Always disable API control when done
    client.enableApiControl(False, vehicle_name=drone1_name)
    client.enableApiControl(False, vehicle_name=drone2_name)
    print("API control disabled for both drones.")