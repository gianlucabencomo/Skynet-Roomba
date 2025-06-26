# Environments

This directory contains the simulation environments, MuJoCo XML files, and wrappers used for training and evaluating Roomba agents.

## Contents

- **sumo_v1.py**: Main Sumo environment implementation (multi-agent, PettingZoo-compatible).
- **wrappers.py**: Environment wrappers (e.g., frame stacking for RL agents).
- **mujoco_env.py**: Base class for MuJoCo-based environments.
- **roomba/**: Contains MuJoCo XML files and assets for different arena and sensor configurations.

## XML Modes

The `roomba/` subdirectory provides four main XML files, each corresponding to a different sensor setup:

- `bump_v1.xml` (`b`): Bump sensors only
- `bump_range_v1.xml` (`br`): Bump + Range sensors
- `bump_range_cliff_v1.xml` (`brc`): Bump + Range + Cliff sensors
- `uwb_v1.xml` (`uwb`): Ultra Wide Band (UWB) localization sensors

These modes match the real-world hardware options available on your Roombas. You can select the mode by passing the `mode` argument to the environment (e.g., `Sumo(mode="brc")`).

## Customization

- **Reward Function**: To modify how agents are rewarded, edit the reward logic in `sumo_v1.py`.
- **Adding New Environments**: Add new MuJoCo XMLs to `roomba/` and create a corresponding Python wrapper if needed.

## Usage Example

```python
from environments.sumo_v1 import Sumo

env = Sumo(mode="uwb")  # Use UWB sensors
```

---
For more details, see code comments in each script and XML file. 