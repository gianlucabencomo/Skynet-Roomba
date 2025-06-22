# Server Directory

This folder contains code for interfacing between the simulation/training environment and real Roomba robots, as well as utilities for manual and neural control.

## Contents

- **joystick.py**: Enables manual control of Roombas using a PS4 controller. Sends commands over the network to the robots. Useful for testing, teleoperation, and debugging.
- **neural.py**: Runs a trained neural agent in real time, sending actions to a physical Roomba based on live sensor data. Supports switching between manual and neural control.
- **constants.py**: Stores network addresses, port numbers, and key mappings for controllers and robots.
- **helper.py**: Utility functions for encoding wheel commands and loading models.
- **position_buffer.py**: (If used) Buffers and processes position data, e.g., from UWB or other tracking systems.

## Usage

- To manually control a Roomba with a PS4 controller:
  ```bash
  python joystick.py --pico1 <roomba1_id> --pico2 <roomba2_id>
  ```
- To run a trained neural agent on a real Roomba:
  ```bash
  python neural.py
  ```
  (Make sure to update the checkpoint path and network settings as needed.)

---
For more details, see code comments in each script. 