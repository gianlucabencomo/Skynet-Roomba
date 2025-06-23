import threading, time, serial
from dataclasses import dataclass
from typing import Dict, Tuple
from constants import SERIAL_PORT, BAUD

@dataclass
class Position:
    x: float; y: float; z: float; t: float

class PositionBuffer:
    def __init__(self, max_age=1.0):
        self._pos: Dict[str, Position] = {}
        self._lock = threading.Lock()
        self._max_age = max_age
    def update(self, tag, x, y, z):
        with self._lock:
            self._pos[tag] = Position(x, y, z, time.time())
    def get_xyt(self):
        now = time.time()
        with self._lock:
            return {t: (p.x, p.y, p.t)
                    for t, p in self._pos.items()
                    if now - p.t < self._max_age}

def enter_shell(ser):
    """
    Kick the module into UART shell mode and wait for the 'dwm>' prompt.
    """
    ser.write(b"\r\r")               # ENTER, ENTER (CR CR)
    ser.flush()
    time.sleep(0.1)                 # small gap (<1 s is OK)
    # Optionally read until we see 'dwm>' prompt:
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if b"dwm" in ser.read(ser.in_waiting or 1):
            return

# ---------- background reader ----------
def reader(buf: PositionBuffer):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.2)
        enter_shell(ser)             # *** important ***
        ser.write(b"lec 1\r")        # NOT \r\n
    except serial.SerialException as e:
        print("cannot open", e)
        return

    while True:
        line = ser.readline()        # returns b'' every 0.2 s if idle
        if not line.startswith(b"POS,"):
            continue
        try:
            _, _, _, tag, xs, ys, zs, *_ = line.decode().split(",")
            buf.update(tag, float(xs), float(ys), float(zs))
        except ValueError:
            continue                 # malformed packet → skip

def main():
    buf = PositionBuffer()
    threading.Thread(target=reader, args=(buf,), daemon=True).start()
    while True:
        for tag, (x, y, _) in buf.get_xyt().items():
            print(f"{tag}: {x:8.2f} {y:8.2f}")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
