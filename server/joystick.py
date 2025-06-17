import socket
import pygame
import time

from helper import encode_wheels, clamp
from constants import *

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))

pygame.init()
joysticks = []
for i in range(pygame.joystick.get_count()):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()
    joysticks.append(joystick)
analog_keys = {0: 0, 1: 0, 2: 0, 3: 0}
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                analog_keys[event.axis] = event.value

        l_vertical = analog_keys[1] # forward / backward
        r_horizontal = analog_keys[2] # left / right

        left = int(100 * clamp(l_vertical - r_horizontal))
        right = int(100 * clamp(l_vertical + r_horizontal))

        # set to 0 if < 30% power
        left = 0 if abs(left) < 30 else left
        # set to 0 if < 30% power
        right = 0 if abs(right) < 30 else right    
        #data, _ = server.recvfrom(8)
        #if data:
        #    print(data.decode())
        command = encode_wheels(right, left)
        print(command)
        server.sendto(command.encode(), (PICO_IP, PORT))
        time.sleep(0.05) # 20 Hz
except KeyboardInterrupt:
    print("Closing connection.")
finally:
    server.close()