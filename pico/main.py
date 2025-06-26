import network
import socket
import utime
from machine import Pin, PWM
from constants import *


def init_motors():
    f1 = PWM(Pin(16, Pin.OUT))  # Right
    b1 = PWM(Pin(17, Pin.OUT))  # Right
    f2 = PWM(Pin(18, Pin.OUT))  # Left
    b2 = PWM(Pin(19, Pin.OUT))  # Left
    for m in (f1, b1, f2, b2):
        m.freq(PWM_FREQ)
    return f1, b1, f2, b2


def set_speed(f, b, v):  # v ∈ [-1,1]
    if v < 0:
        f.duty_u16(0)
        b.duty_u16(int(MAX_DUTY_U16 * -v))
    else:
        b.duty_u16(0)
        f.duty_u16(int(MAX_DUTY_U16 * v))


f1, b1, f2, b2 = init_motors()

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
while not wlan.isconnected():
    utime.sleep(0.1)
print("Wi-Fi connected:", wlan.ifconfig()[0])

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.bind((HOST, PORT))

try:
    while True:
        data, addr = client.recvfrom(8)  # blocking until command arrives
        if not data:
            continue
        cmd = data.decode()
        v_R = int(cmd[0:3]) * (-1 if cmd[3] == "-" else 1) / 100
        v_L = int(cmd[4:7]) * (-1 if cmd[7] == "-" else 1) / 100
        set_speed(f1, b1, v_R)
        set_speed(f2, b2, v_L)

except KeyboardInterrupt:
    print("Shutting down.")
finally:
    for m in (f1, b1, f2, b2):
        m.duty_u16(0)
    client.close()
