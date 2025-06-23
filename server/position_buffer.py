import threading, time, re, serial
from dataclasses import dataclass
from typing import Dict, Optional
from constants import SERIAL_PORT, BAUD

#  listenerID, tagID, x, y, z, quality, crc
_POS_RE = re.compile(
    r"POS,(\d+),(\d+),([-0-9.]+),([-0-9.]+),([-0-9.]+),(\d+),x[0-9A-Fa-f]+"
)

@dataclass
class Position:
    x: float 
    y: float
    z: float
    timestamp: float

class PositionBuffer:
    def __init__(self, max_age=1.0):
        self._pos: Dict[int, Position] = {}
        self._lock = threading.Lock()
        self._max_age = max_age

    def update(self, tag: int, x: float, y: float, z: float):
        with self._lock:
            self._pos[tag] = Position(x, y, z, time.time())

    def get(self, tag: int) -> Optional[Position]:
        with self._lock:
            p = self._pos.get(tag)
            return p if p and time.time() - p.timestamp < self._max_age else None

    def get_xyt(self) -> Dict[int, Tuple[float, float, float]]:
        """Returns x, y, timestamp for each tag in the buffer."""
        with self._lock:
            return {
                tag_id: (pos.x, pos.y, pos.timestamp)
                for tag_id, pos in self._pos.items()
            }

def reader(buf: PositionBuffer):
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    for raw in iter(lambda: ser.readline().decode(errors="ignore").strip(), ""):
        m = _POS_RE.match(raw)
        if not m:
            continue
        _listener, tag_dec, x, y, z, _q = m.groups()
        buf.update(int(tag_dec), float(x), float(y), float(z))

def test():
    buf = PositionBuffer()
    threading.Thread(target=reader, args=(buf,), daemon=True).start()

    while True:
        xyt = buf.get_xyt()
        for tag_id, (x, y, t) in xyt.items():
            print(f"Tag {tag_id}: x={x:.2f}, y={y:.2f}, t={t:.2f}")
        print("-" * 40)
        time.sleep(0.1)

if __name__ == "__main__":
    test()
