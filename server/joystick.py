import argparse
import pygame
import socket
import time
from helper import encode_wheels, clamp
from constants import *

def run_joystick(pico_ip: str):
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((HOST, PORT))
    server.settimeout(0.01)

    pygame.init()
    joysticks = []
    for i in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)
    analog_keys = {0: 0, 1: 0, 2: 0, 3: 0}
    tricks = {"circle_left": False, "circle_right": False}
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    analog_keys[event.axis] = event.value

                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == PS4_KEYS['x']:
                        tricks["circle_left"] = not tricks["circle_left"]
                        tricks["circle_right"] = False
                    elif event.button == PS4_KEYS['circle']:
                        tricks["circle_left"] = False
                        tricks["circle_right"] = not tricks["circle_right"]

            if tricks["circle_left"]:
                left, right = 50, 100
            elif tricks["circle_right"]:
                left, right = 100, 50
            else:
                l_vertical = analog_keys[1] # forward / backward
                r_horizontal = analog_keys[2] # left / right

                left = int(100 * clamp(l_vertical - r_horizontal))
                right = int(100 * clamp(l_vertical + r_horizontal))

                # set to 0 if < DEADZONE% power
                left = 0 if abs(left) < DEADZONE else left
                # set to 0 if < DEADZONE% power
                right = 0 if abs(right) < DEADZONE else right    

            command = encode_wheels(right, left)
            print(f'Command : {command}')
            server.sendto(command.encode(), (pico_ip, PORT))
            time.sleep(0.05) # 20 Hz
    except KeyboardInterrupt:
        print("Closing connection.")
    finally:
        server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pico", choices=PICO_IPS.keys(), default="base")
    args = parser.parse_args()
    run_joystick(PICO_IPS[args.pico])