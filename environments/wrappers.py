import numpy as np
from collections import deque
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Box


class FrameStackWrapper(BaseParallelWrapper):
    """
    Frame-stacking for PettingZoo ParallelEnvs that is compatible with
    `pettingzoo_env_to_vec_env_v1` and Supersuit vectorisation.

    Parameters
    ----------
    env : ParallelEnv
        Environment to wrap.
    k : int
        Number of recent frames to concatenate.
    concat_axis : int
        Axis along which frames are concatenated (0 for 1-D feature vectors,
        -1 for image channels, etc.).
    """

    def __init__(self, env, k: int = 4, concat_axis: int = 0):
        super().__init__(env)
        assert k > 0, "`k` (stack size) must be >= 1"
        self.k = k
        self.concat_axis = concat_axis

        # One deque per agent (filled on reset)
        self._frames = {agent: deque(maxlen=k) for agent in self.possible_agents}

        # ── Build the stacked observation spaces ───────────────────────────
        self._obs_spaces = {}
        for agent, space in env.observation_spaces.items():
            low  = np.concatenate([space.low]  * k, axis=concat_axis)
            high = np.concatenate([space.high] * k, axis=concat_axis)
            self._obs_spaces[agent] = Box(low=low, high=high, dtype=space.dtype)

    # ------------------------------------------------------------------ #
    #  Interfaces PettingZoo & Supersuit expect                          #
    # ------------------------------------------------------------------ #
    @property
    def observation_spaces(self):
        # PettingZoo’s dict-style access
        return self._obs_spaces

    def observation_space(self, agent):
        # PettingZoo still calls this *function* during vectorisation
        return self._obs_spaces[agent]

    # ------------------------------------------------------------------ #
    #  Core API                                                          #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        for agent, ob in obs.items():
            dq = self._frames[agent]
            dq.clear()
            for _ in range(self.k):              # pre-fill with first obs
                dq.append(ob.copy())
            obs[agent] = self._stack(agent)
        return obs, infos

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        for agent, ob in obs.items():
            self._frames[agent].append(ob.copy())
            obs[agent] = self._stack(agent)
        return obs, rews, terms, truncs, infos

    # ------------------------------------------------------------------ #
    #  Helper                                                            #
    # ------------------------------------------------------------------ #
    def _stack(self, agent):
        """Concatenate the `k` most recent frames for `agent`."""
        return np.concatenate(list(self._frames[agent]), axis=self.concat_axis)
