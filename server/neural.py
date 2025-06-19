import socket
import pygame
import time

import torch

from helper import encode_wheels, clamp, load_checkpoint
from constants import *

from collections import deque

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind((HOST, PORT))
server.settimeout(0.01)

agent = load_checkpoint("./checkpoints/agent_roomba__0__1750199231_train_step_5242880_timestamp_1750199231.pt", 80, 2, "cpu")

pygame.init()
joysticks = []
for i in range(pygame.joystick.get_count()):
    joystick = pygame.joystick.Joystick(i)
    joystick.init()
    joysticks.append(joystick)
analog_keys, neural = {0: 0, 1: 0, 2: 0, 3: 0}, False
observations = deque([torch.zeros(OBS_DIM) for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
left, right = 0., 0.
da, db = 4., 4.
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                analog_keys[event.axis] = event.value

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == PS4_KEYS['x']:
                    neural = not neural

        try:
            data, _ = server.recvfrom(16)  # Pico sends "123.4,456.7"
            da, db = map(float, data.decode().split(","))
            print(f"Front-A {da:5.1f} cm | Front-B {db:5.1f} cm")
        except socket.timeout:
            pass  # nothing yet
        obs = torch.tensor([left / 100., right / 100., da / 100., db / 100.])
        observations.append(obs)
            
        if neural:
            with torch.no_grad():
                action, *_ = agent.get_action_and_value(torch.stack(list(observations)).view(1, -1))
            left, right = action.cpu().numpy().squeeze()
        else:
            l_vert = analog_keys[1]
            r_horz = analog_keys[2]
            left, right = l_vert - r_horz, l_vert + r_horz

        left  = int(100 * clamp(left))
        right = int(100 * clamp(right))
        # dead-band 30 %
        left  = 0 if abs(left)  < 30 else left
        right = 0 if abs(right) < 30 else right

        cmd = encode_wheels(right, left)
        server.sendto(cmd.encode(), (PICO_IP, PORT))
        time.sleep(0.099)  # 10 Hz command rate
except KeyboardInterrupt:
    print("Closing connection.")
finally:
    server.close()