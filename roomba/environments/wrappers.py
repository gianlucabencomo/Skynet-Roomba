import numpy as np
from collections import deque
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Box
from typing import List, Callable

class FrameStackWrapper(BaseParallelWrapper):
    """FrameStack Wrapper for MLP network to have explicit memory."""

    def __init__(self, env, k: List[int] = [10, 10], concat_axis: int = 0):
        super().__init__(env)
        assert len(k) == len(env.possible_agents), "k must match length of possible agents"
        self.k = k
        self.concat_axis = concat_axis
        self._frames = {agent: deque(maxlen=k[i]) for i, agent in enumerate(self.possible_agents)}
        self._obs_spaces = {}
        for i, (agent, space) in enumerate(env.observation_spaces.items()):
            low = np.concatenate([space.low] * k[i], axis=concat_axis)
            high = np.concatenate([space.high] * k[i], axis=concat_axis)
            self._obs_spaces[agent] = Box(low=low, high=high, dtype=space.dtype)

    @property
    def observation_spaces(self):
        return self._obs_spaces

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        for i, (agent, ob) in enumerate(obs.items()):
            dq = self._frames[agent]
            dq.clear()
            for _ in range(self.k[i]):
                dq.append(ob.copy())
            obs[agent] = self._stack(agent)
        return obs, infos

    def step(self, actions):
        obs, rews, terms, truncs, infos = self.env.step(actions)
        for agent, ob in obs.items():
            self._frames[agent].append(ob.copy())
            obs[agent] = self._stack(agent)
        return obs, rews, terms, truncs, infos

    def _stack(self, agent):
        return np.concatenate(list(self._frames[agent]), axis=self.concat_axis)

class DomainRandomizationWrapper(BaseParallelWrapper):
    """Fixed Domain Randomization Wrapper for Sumo."""
    def __init__(self, env, 
                 obs_alpha_sample: Callable[[None], float] = lambda _: np.random.uniform(low=0.6, high=0.95, size=()), 
                 uwb_sensor_noise_sample: Callable[[None], float] = lambda _: np.random.uniform(low=0.0, high=0.2, size=())):
        super().__init__(env)
        self.obs_alpha_sample = obs_alpha_sample
        self.uwb_sensor_noise_sample = uwb_sensor_noise_sample

    @property
    def observation_spaces(self):
        return self.env.observation_spaces

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def reset(self, seed=None, options=None):
        new_obs_alpha = self.obs_alpha_sample(None)
        new_sensor_noise = self.uwb_sensor_noise_sample(None)
        self.env._obs_alpha = new_obs_alpha
        self.env._uwb_sensor_noise = new_sensor_noise

        return self.env.reset(seed=seed, options=options)