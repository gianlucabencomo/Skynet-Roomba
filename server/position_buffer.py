import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class Position:
    x: float
    y: float
    z: float
    timestamp: float

class PositionBuffer:
    """Thread-safe buffer for storing the latest position data for multiple tags."""
    
    def __init__(self, max_age_seconds: float = 1.0):
        self._positions: Dict[str, Position] = {}
        self._lock = threading.Lock()
        self._max_age = max_age_seconds
    
    def update_position(self, tag_id: str, x: float, y: float, z: float) -> None:
        """Update position for a specific tag ID."""
        with self._lock:
            self._positions[tag_id] = Position(x, y, z, time.time())
    
    def get_position(self, tag_id: str) -> Optional[Position]:
        """Get the latest position for a tag ID, returns None if too old or not found."""
        with self._lock:
            if tag_id not in self._positions:
                return None
            
            position = self._positions[tag_id]
            if time.time() - position.timestamp > self._max_age:
                return None
            
            return position
    
    def get_positions(self, tag_ids: list) -> Dict[str, Optional[Position]]:
        """Get positions for multiple tag IDs."""
        return {tag_id: self.get_position(tag_id) for tag_id in tag_ids}
    
    def get_relative_position(self, tag1_id: str, tag2_id: str) -> Optional[Tuple[float, float]]:
        """Get relative position (x, y) from tag1 to tag2."""
        pos1 = self.get_position(tag1_id)
        pos2 = self.get_position(tag2_id)
        
        if pos1 is None or pos2 is None:
            return None
        
        return (pos2.x - pos1.x, pos2.y - pos1.y)
    
    def is_data_fresh(self, tag_id: str) -> bool:
        """Check if data for tag is fresh (within max_age)."""
        return self.get_position(tag_id) is not None

# Global position buffer instance
position_buffer = PositionBuffer() 