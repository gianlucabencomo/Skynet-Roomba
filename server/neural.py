import socket
import pygame
import time
import numpy as np
import threading
import serial

import torch

from helper import encode_wheels, clamp, load_checkpoint
from constants import *
from position_buffer import position_buffer

# Configuration
SERIAL_PORT = '/dev/tty.usbmodem0007601851681'  # Update this to your actual port
SERIAL_BAUDRATE = 115200

def run_position_reader():
    """Run the position reader in a background thread."""
    
    try:
        # Open serial connection
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"Position reader connected to {SERIAL_PORT} at {SERIAL_BAUDRATE} baud")
        
        # Wait a moment for connection to stabilize
        time.sleep(2)
        
        # Send initialization commands
        ser.write(b'\n\n')
        ser.write(b'lec\n')
        
        print(f"Position reader tracking tag IDs: {ROOMBA_1_TAG_ID}, {ROOMBA_2_TAG_ID}")
        
        # Read and process the stream
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and line.startswith('POS'):
                try:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        tag_id = parts[2]
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                        
                        # Update position buffer for tracked tags
                        if tag_id in [ROOMBA_1_TAG_ID, ROOMBA_2_TAG_ID]:
                            position_buffer.update_position(tag_id, x, y, z)
                            print(f"Position - Tag {tag_id}: x: {x:.2f}m, y: {y:.2f}m, z: {z:.2f}m")
                        
                except (ValueError, IndexError) as e:
                    print(f"Position reader parse error: {line} - {e}")
            
            time.sleep(0.01)
            
    except serial.SerialException as e:
        print(f"Position reader serial error: {e}")
    except Exception as e:
        print(f"Position reader error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Position reader serial connection closed")

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))
server.settimeout(0.01)

agent = load_checkpoint("./checkpoints/agent_roomba__0__1750181679_train_step_5242880_timestamp_1750181679.pt", 11, 2, "cpu")

pygame.init()
joysticks = []
for i in range(pygame.joystick.get_count()):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()
    joysticks.append(joystick)

analog_keys, neural = {0: 0, 1: 0, 2: 0, 3: 0}, False
current_torques = [0.0, 0.0]  # Track current wheel torques for observation

def create_observation():
    """Create 11-dimensional observation vector based on current state."""
    # Get positions from buffer
    my_pos = position_buffer.get_position(ROOMBA_1_TAG_ID)
    opponent_pos = position_buffer.get_position(ROOMBA_2_TAG_ID)
    
    # Default values if position data is not available
    if my_pos is None:
        my_xy = np.array([0.0, 0.0])
        print("Warning: No position data for roomba 1")
    else:
        my_xy = np.array([my_pos.x, my_pos.y])
    
    if opponent_pos is None:
        opponent_xy = np.array([0.0, 0.0])
        print("Warning: No position data for roomba 2")
    else:
        opponent_xy = np.array([opponent_pos.x, opponent_pos.y])
    
    # Calculate observation components (matching sumo_v3.py structure)
    # Distance to center
    my_dist = np.linalg.norm(my_xy)
    
    # Current torques (normalized to [-1, 1] range)
    torque_left = current_torques[0] / 100.0
    torque_right = current_torques[1] / 100.0
    
    # Relative position and velocity (velocity approximated as zero for now)
    rel_pos = my_xy - opponent_xy
    rel_vel = np.array([0.0, 0.0])  # Would need position history to calculate
    
    # Construct 11-dimensional observation
    obs = np.concatenate([
        np.array([my_dist, torque_left, torque_right]),  # 3 dims
        rel_pos,  # 2 dims
        rel_vel,  # 2 dims
        my_xy,    # 2 dims - adding absolute position
        opponent_xy  # 2 dims - adding opponent absolute position
    ])
    
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

# Start position reader in background thread
print("Starting Roomba Neural Control with Position Tracking")
print("=" * 60)
reader_thread = threading.Thread(target=run_position_reader, daemon=True)
reader_thread.start()

# Give reader time to initialize
time.sleep(3)

try:
    while True:
        print(f"Time: {time.time():.2f}, Neural: {neural}")
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                analog_keys[event.axis] = event.value

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == PS4_KEYS['x']:
                    neural = not neural
                    print(f"Switched to {'NEURAL' if neural else 'MANUAL'} control")
            
            if neural:
                # Create observation from position data
                obs = create_observation()
                with torch.no_grad():
                    action, *_ = agent.get_action_and_value(obs)
                left, right = action.cpu().numpy().squeeze()
            else:
                l_vert = analog_keys[1]
                r_horz = analog_keys[2]
                left, right = l_vert - r_horz, l_vert + r_horz

        # Convert to motor commands
        left  = int(100 * clamp(left))
        right = int(100 * clamp(right))
        
        # Update current torques for next observation
        current_torques = [left, right]
        
        # dead-band 30 %
        left  = 0 if abs(left)  < 30 else left
        right = 0 if abs(right) < 30 else right

        cmd = encode_wheels(right, left)
        server.sendto(cmd.encode(), (PICO_IP, PORT))

        try:
            data, _ = server.recvfrom(16)  # Pico sends "123.4,456.7"
            da, db = map(float, data.decode().split(","))
            print(f"Front-A {da:5.1f} cm | Front-B {db:5.1f} cm")
        except socket.timeout:
            pass  # nothing yet
        time.sleep(0.099)  # 10 Hz command rate
except KeyboardInterrupt:
    print("Closing connection.")
finally:
    server.close()