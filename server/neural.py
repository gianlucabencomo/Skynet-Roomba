import argparse, socket, time, threading
import pygame, torch, numpy as np

from helper import encode_wheels, clamp, load_checkpoint, build_obs
from constants import *
from state_buffer import StateBuffer, State, reader
from collections import deque

MAXIMUS_TAG = "5620"
COMMODUS_TAG = "4F2A"
POLICY_MAX_PATH = "./test_ckpt.pt"
POLICY_COM_PATH = "./test_ckpt.pt"
MAX_ALPHA = 0.2
COM_ALPHA = 0.2
MAX_FRAME_STACK = 10
COM_FRAME_STACK = 10

policy_max = load_checkpoint(
    POLICY_MAX_PATH, obs_dim=100, action_dim=2, actor_hidden_widths=[128, 128], critic_hidden_widths=[128, 128], device="cpu"
).eval()
policy_com = load_checkpoint(
    POLICY_COM_PATH, obs_dim=100, action_dim=2, actor_hidden_widths=[128, 128], critic_hidden_widths=[128, 128], device="cpu"
).eval()

last_torque_max = np.zeros(2, dtype=np.float32)
last_torque_com = np.zeros(2, dtype=np.float32)


def run_joystick(pico_ip1: str, pico_ip2: str):
    # ---- UDP + joystick init -------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(0.01)

    pygame.init()
    joy = pygame.joystick.Joystick(0)
    joy.init()
    analog = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

    # ---- state-buffer thread -------------------------------------------------
    buf = StateBuffer()
    threading.Thread(target=reader, args=(buf,), daemon=True).start()

    active_pico = pico_ip1
    running = True
    neural_mode = False

    global last_torque_max, last_torque_com

    obs_max = deque(maxlen=MAX_FRAME_STACK)
    obs_com = deque(maxlen=COM_FRAME_STACK)

    try:
        while running:
            # ---------- pygame events ----------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    analog[event.axis] = event.value
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == PS4_KEYS["x"]:
                        neural_mode = not neural_mode
                    elif event.button == PS4_KEYS["triangle"]:
                        active_pico = pico_ip2 if active_pico == pico_ip1 else pico_ip1
                    elif event.button == PS4_KEYS["square"]:
                        # emergency stop
                        for ip in (pico_ip1, pico_ip2):
                            sock.sendto(encode_wheels(0, 0).encode(), (ip, PORT))
                        running = False

            if neural_mode:
                max_s = buf.get(MAXIMUS_TAG)
                com_s = buf.get(COMMODUS_TAG)

                if (max_s is not None) and (com_s is not None):
                    obs = build_obs(max_s, last_torque_max, com_s, last_torque_com)
                    obs_max.append(obs["maximus"])
                    obs_com.append(obs["commodus"])
                    if len(obs_max) == 1:
                        for _ in range(MAX_FRAME_STACK):
                            obs_max.append(obs["maximus"])
                        for _ in range(COM_FRAME_STACK):
                            obs_com.append(obs["commodus"])

                    com_stacked_obs = np.concatenate(obs_com, axis=0).reshape(1, -1)
                    com_obs_tensor = torch.from_numpy(com_stacked_obs).float()
                    max_stacked_obs = np.concatenate(obs_max, axis=0).reshape(1, -1)
                    max_obs_tensor = torch.from_numpy(max_stacked_obs).float()
                    with torch.no_grad():
                        torque_com, *_ = policy_com.get_action_and_value(com_obs_tensor)
                        torque_max, *_ = policy_max.get_action_and_value(max_obs_tensor)
                        torque_com = -torque_com.squeeze().cpu().numpy() # sign flipped in real life
                        torque_max = -torque_max.squeeze().cpu().numpy() # sign flipped in real life

                    # EMA smoothing
                    torque_max = (
                        MAX_ALPHA * torque_max + (1 - MAX_ALPHA) * last_torque_max
                    )
                    torque_com = (
                        COM_ALPHA * torque_com + (1 - COM_ALPHA) * last_torque_com
                    )
                    last_torque_max[:] = torque_max
                    last_torque_com[:] = torque_com

                    # map to PWM
                    max_left = int(100 * clamp(torque_max[0]))
                    max_right = int(100 * clamp(torque_max[1]))
                    com_left = int(100 * clamp(torque_com[0]))
                    com_right = int(100 * clamp(torque_com[1]))
                else:
                    # no fresh UWB → stop both
                    max_left = max_right = com_left = com_right = 0
            # ---------- MANUAL mode ------------------------------------------
            else:
                fwd = analog[1]  # left stick vertical
                turn = analog[2]  # right stick horizontal
                l_pwm = int(100 * clamp(fwd - turn))
                r_pwm = int(100 * clamp(fwd + turn))
                cl_pwm = cr_pwm = 0  # only controlling active pico

            # ---------- Apply deadband ---------------------------------------
            l_pwm = 0 if abs(l_pwm) < DEADBAND else l_pwm
            r_pwm = 0 if abs(r_pwm) < DEADBAND else r_pwm
            cl_pwm = 0 if abs(cl_pwm) < DEADBAND else cl_pwm
            cr_pwm = 0 if abs(cr_pwm) < DEADBAND else cr_pwm

            # ---------- Send commands ---------------------------------------
            if neural_mode:
                # send to both Picos
                sock.sendto(
                    encode_wheels(max_left, max_right).encode(), (pico_ip1, PORT)
                )
                sock.sendto(
                    encode_wheels(com_left, com_right).encode(), (pico_ip2, PORT)
                )
                print(
                    f"NN → {pico_ip1}: {l_pwm:+3d}/{r_pwm:+3d}   {pico_ip2}: {cl_pwm:+3d}/{cr_pwm:+3d}"
                )
            else:
                # manual → only active pico
                cmd = encode_wheels(r_pwm, l_pwm)
                sock.sendto(cmd.encode(), (active_pico, PORT))
                print(f"MAN → {active_pico}: {cmd}")

            time.sleep(0.1)
    finally:
        sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pico1", choices=PICO_IPS.keys(), default="base")
    ap.add_argument("--pico2", choices=PICO_IPS.keys(), default="base")
    args = ap.parse_args()
    run_joystick(PICO_IPS[args.pico1], PICO_IPS[args.pico2])
