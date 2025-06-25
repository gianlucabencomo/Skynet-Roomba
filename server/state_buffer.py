import threading, time, serial
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from constants import SERIAL_PORT, BAUD


@dataclass
class State:
    x: float
    y: float
    vx: float
    vy: float
    dist: float
    t: float


class StateBuffer:
    def __init__(self, origin: Tuple[float, float] = (1.8, 1.75), max_age: float = 1.0):
        self._state: Dict[str, State] = {}
        self._lock = threading.Lock()
        self._origin = origin
        self._max_age = max_age

    def update(self, tag: str, x: float, y: float):
        with self._lock:
            now = time.time()
            if tag in self._state:
                prev = self._state[tag]
                dt = now - prev.t
                vx = (x - prev.x) / dt if dt > 1e-3 else 0.0
                vy = (y - prev.y) / dt if dt > 1e-3 else 0.0
            else:
                vx = vy = 0.0
            dist = np.linalg.norm([x - self._origin[0], y - self._origin[1]])
            self._state[tag] = State(x, y, vx, vy, dist, now)

    def get(self, tag: str) -> State:
        return self._state.get(tag)


def enter_shell(ser):
    ser.write(b"\r\r")
    ser.flush()
    time.sleep(0.1)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if b"dwm" in ser.read(ser.in_waiting or 1):
            return


def reader(buf: StateBuffer):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
        enter_shell(ser)
        ser.write(b"lec 1\r")
    except serial.SerialException as e:
        print("cannot open", e)
        return

    while True:
        line = ser.readline()
        if not line.startswith(b"POS,"):
            continue
        try:
            _, _, _, tag, xs, ys, *_ = line.decode(errors="ignore").split(",")
            buf.update(tag, float(xs), float(ys))
        except ValueError:
            continue


def main():
    buf = StateBuffer()
    threading.Thread(target=reader, args=(buf,), daemon=True).start()
    while True:
        for tag, state in buf.get().items():
            print(
                f"{tag:>8} | x={state.x:6.2f} y={state.y:6.2f} vx={state.vx:5.2f} vy={state.vy:5.2f} speed={state.dist:5.2f}"
            )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
