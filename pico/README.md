# Pico

This folder contains firmware and configuration code for the microcontroller (e.g., Raspberry Pi Pico 2 W) mounted on the Roomba robot.

## Purpose

- Receives wheel commands over Wi-Fi from the server/controller.
- Decodes and applies commands to the Roomba's motors for real-time control.
- Bridges the gap between high-level agent logic (simulation or neural) and low-level hardware actuation.

## Contents

- **main.py**: Main firmware script. Listens for UDP packets, decodes wheel commands, and sets motor speeds accordingly.
- **constants.py**: Wi-Fi credentials, network settings, and hardware constants for the microcontroller.

## Usage

1. Flash `main.py` and `constants.py` to your Pico W (or compatible microcontroller).
2. Update `constants.py` with your Wi-Fi SSID, password, and network settings.
3. Power on the Pico and ensure it connects to your Wi-Fi network.
4. The Pico will listen for UDP commands from the server (see `server/joystick.py` or `server/neural.py`).

## Integration

- The Pico code is designed to work seamlessly with the server scripts in the `server/` directory.
- Make sure network addresses and ports match between the Pico and server configurations.

---
For more details, see code comments in each script.