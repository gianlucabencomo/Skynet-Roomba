import argparse
import pygame
import socket
import time
from helper import encode_wheels, clamp
from constants import *
from typing import List

def run_joystick(picos: List[str], n_joysticks: int = 1):
    # -- initial checks --
    if n_joysticks not in [1, 2]:
        raise ValueError("Number of joysticks must be either 1 or 2.")
    if n_joysticks < len(picos):
        raise ValueError("Number of robots is less than number of controllers.")
    for pico in picos:
        if pico not in PICO_IPS:
            raise ValueError(f"Unknown robot specified: {pico}")
    
    # -- set up server --
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((HOST, PORT))
    server.settimeout(0.01)

    # -- init joysticks --
    pygame.init()
    available = pygame.joystick.get_count()
    if available < n_joysticks:
        raise RuntimeError(f"Expected {n_joysticks} joystick(s), but only found {available}.")
    
    joysticks = []
    for i in range(n_joysticks):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)
    analog_keys = [{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(n_joysticks)]
    running = True
    try:
        while running:
            for event in pygame.event.get():
                jid = event.joy # joystick id
                if event.type == pygame.JOYAXISMOTION:
                    analog_keys[jid][event.axis] = event.value

                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == PS4_KEYS["x"]:
                        pass
                    elif event.button == PS4_KEYS["circle"]:
                        pass
                    if event.button == PS4_KEYS["triangle"]:
                        # -- switch active picos for player(s) --
                        picos = picos[1:] + picos[:1]
                    if event.button == PS4_KEYS["square"]:
                        # -- kill --
                        for pico in picos:
                            server.sendto(encode_wheels(0, 0).encode(), (PICO_IPS[pico], PORT))
                        running = False
            for i, keys in enumerate(analog_keys):
                pico = picos[i]
                l_vertical = keys[1]  # forward / backward
                r_horizontal = keys[2]  # left / right

                left = int(100 * clamp(l_vertical - r_horizontal))
                right = int(100 * clamp(l_vertical + r_horizontal))

                # set to 0 if < DEADZONE% power
                left = 0 if abs(left) < DEADBAND else left
                # set to 0 if < DEADZONE% power
                right = 0 if abs(right) < DEADBAND else right

                command = encode_wheels(right, left)
                print(f"{pico} : {command}")
                server.sendto(command.encode(), (PICO_IPS[pico], PORT))
            time.sleep(0.05)  # 20 Hz
    except KeyboardInterrupt:
        print("Closing connection.")
    finally:
        server.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--picos", nargs='+', choices=PICO_IPS.keys(), required=True, help="List of pico names (space separated)")
    parser.add_argument("--n_joysticks", type=int, required=True, help="Number of joysticks to initialize")
    args = parser.parse_args()
    run_joystick(args.picos, args.n_joysticks)
