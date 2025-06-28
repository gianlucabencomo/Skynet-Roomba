__credits__ = ["Gianluca"]

from typing import Dict, Tuple, Union, Optional

import os, warnings
import numpy as np

import mujoco

from pettingzoo import ParallelEnv

from gymnasium import utils
from .mujoco_env import MujocoEnv
from gymnasium.spaces import Box

from collections import deque

# -- file paths --
CURRENT_DIR = os.path.dirname(__file__)
BUMP_PATH = os.path.join(CURRENT_DIR, "roomba", "bump_v1.xml")
BUMP_RANGE_PATH = os.path.join(CURRENT_DIR, "roomba", "bump_range_v1.xml")
BUMP_RANGE_CLIFF_PATH = os.path.join(CURRENT_DIR, "roomba", "bump_range_cliff_v1.xml")
UWB_PATH = os.path.join(CURRENT_DIR, "roomba", "uwb_v1.xml")
PATHS = {
    "uwb": UWB_PATH,
    "b": BUMP_PATH,
    "br": BUMP_RANGE_PATH,
    "brc": BUMP_RANGE_CLIFF_PATH,
}

# -- camera --
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 0.2)),
    "elevation": -30.0,
}


class Sumo(ParallelEnv, MujocoEnv, utils.EzPickle):
    metadata = {
        "name": "sumo",
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        frame_skip: int = 50,  # sim @ 500hz, actions @ 10hz
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        match_length: float = 60.0,  # 1-minute matches
        reset_noise_scale: float = 1e-2,
        contact_rew_weight: float = 1e-1,  # rew function is win/lose + exploratory that maximizes contact
        symmetry_rew_weight: float = 1e-2,
        dist_center_weight: float = 1e-2,
        uwb_sensor_noise: float = 0.05,  # for sim2real
        deadband: float = 0.1,  # if ctrl < 0.1 then set ctrl = 0
        action_alpha: float = 0.4,  # EMA for actions
        obs_alpha: float = 0.6,  # EMA for obs to replicate UWB filter
        train: bool = True,  # randomly rotate + translate roombas if *not in eval mode
        mode: str = "uwb",
        **kwargs,
    ):
        if mode.lower() not in PATHS:
            warnings.warn(
                f"Unknown mode '{mode}'; falling back to 'uwb'. Valid modes are {list(PATHS.keys())}.",
                stacklevel=2,
            )
            mode = "uwb"
        xml_file = PATHS[mode.lower()]
        utils.EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            match_length,
            reset_noise_scale,
            contact_rew_weight,
            symmetry_rew_weight,
            dist_center_weight,
            uwb_sensor_noise,
            deadband,
            action_alpha,
            obs_alpha,
            train,
            mode,
            **kwargs,
        )
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._match_length = match_length
        self._reset_noise_scale = reset_noise_scale
        self._contact_rew_weight = contact_rew_weight
        self._symmetry_rew_weight = symmetry_rew_weight
        self._dist_center_weight = dist_center_weight
        self._uwb_sensor_noise = uwb_sensor_noise
        self._deadband = deadband
        self._train = train
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self._match_time = 0.0
        self._mode = mode
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._qvel_tm1 = self.data.qvel.copy()
        self._qvel_tm2 = self._qvel_tm1.copy()
        self._action_alpha = action_alpha
        self._obs_alpha = obs_alpha

        self._maximus_pos_history = deque(maxlen=7)
        self._commodus_pos_history = deque(maxlen=7)
        self._maximus_vel_history = deque(maxlen=7)
        self._commodus_vel_history = deque(maxlen=7)

        self.setup_spaces()

    def setup_spaces(self):
        maximus_inds, commodus_inds = [], []
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name is None:
                continue
            if act_name[-1] == "1":
                maximus_inds.append(i)
            elif act_name[-1] == "2":
                commodus_inds.append(i)
        maximus_low = np.array(
            [self.model.actuator_ctrlrange[i, 0] for i in maximus_inds],
            dtype=np.float32,
        )
        maximus_high = np.array(
            [self.model.actuator_ctrlrange[i, 1] for i in maximus_inds],
            dtype=np.float32,
        )
        commodus_low = np.array(
            [self.model.actuator_ctrlrange[i, 0] for i in commodus_inds],
            dtype=np.float32,
        )
        commodus_high = np.array(
            [self.model.actuator_ctrlrange[i, 1] for i in commodus_inds],
            dtype=np.float32,
        )
        self.action_spaces = {
            "maximus": Box(low=maximus_low, high=maximus_high, dtype=np.float32),
            "commodus": Box(low=commodus_low, high=commodus_high, dtype=np.float32),
        }
        self.maximus_actuator_inds = maximus_inds
        self.commodus_actuator_inds = commodus_inds

        obs_dim = self._get_obs()["maximus"].shape[0]
        self.observation_spaces = {
            "maximus": Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            "commodus": Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
        }

    @property
    def possible_agents(self):
        return ["maximus", "commodus"]

    @property
    def agents(self):
        return ["maximus", "commodus"]

    def _get_obs(self):
        # -- torques (as ctrl input) --
        maximus_torqueL, maximus_torqueR = self.data.ctrl[0], self.data.ctrl[1]
        commodus_torqueL, commodus_torqueR = self.data.ctrl[2], self.data.ctrl[3]
        if self._mode == "uwb":
            qpos = self.data.qpos.flatten()
            # -- x, y position (w/ sensor noise if enabled) --
            maximus_xy_pos, commodus_xy_pos = qpos[0:2], qpos[9:11]
            if self._uwb_sensor_noise > 0.0:
                maximus_xy_pos += np.random.normal(
                    loc=0.0, scale=self._uwb_sensor_noise, size=(2,)
                )
                commodus_xy_pos += np.random.normal(
                    loc=0.0, scale=self._uwb_sensor_noise, size=(2,)
                )

            self._maximus_pos_history.append(maximus_xy_pos)
            self._commodus_pos_history.append(commodus_xy_pos)
            maximus_xy_pos = np.mean(self._maximus_pos_history, axis=0)
            commodus_xy_pos = np.mean(self._commodus_pos_history, axis=0)

            if len(self._maximus_pos_history) < 2 or len(self._commodus_pos_history) < 2:
                # Use zeros for velocity if not enough history
                maximus_xy_vel = np.zeros(2, dtype=np.float32)
                commodus_xy_vel = np.zeros(2, dtype=np.float32)
            else:
                maximus_xy_vel = (self._maximus_pos_history[-1] - self._maximus_pos_history[-2]) / self.dt
                commodus_xy_vel = (self._commodus_pos_history[-1] - self._commodus_pos_history[-2]) / self.dt


            self._maximus_vel_history.append(maximus_xy_vel)
            self._commodus_vel_history.append(commodus_xy_vel)

            maximus_xy_vel = np.mean(self._maximus_vel_history, axis=0)
            commodus_xy_vel = np.mean(self._commodus_vel_history, axis=0)

            # -- maximus to commodus relative --
            rel_pos = maximus_xy_pos - commodus_xy_pos
            rel_vel = maximus_xy_vel - commodus_xy_vel

            maximus_obs = np.concatenate(
                [
                    maximus_xy_pos,
                    maximus_xy_vel,
                    [maximus_torqueL, maximus_torqueR],
                    rel_pos,
                    rel_vel,
                ]
            )
            commodus_obs = np.concatenate(
                [
                    commodus_xy_pos,
                    commodus_xy_vel,
                    [commodus_torqueL, commodus_torqueR],
                    -rel_pos,
                    -rel_vel,
                ]
            )
        else:
            # -- pull sensor data from sim and load that as obs --
            half = self.model.nsensor // 2
            sensordata = self.data.sensordata
            maximus_sensor, commodus_sensor = sensordata[:half], sensordata[half:]
            maximus_obs = np.concatenate(
                [np.array([maximus_torqueL, maximus_torqueR]), maximus_sensor]
            )
            commodus_obs = np.concatenate(
                [np.array([commodus_torqueL, commodus_torqueR]), commodus_sensor]
            )

        return {
            "maximus": maximus_obs,
            "commodus": commodus_obs,
        }

    def step(self, actions):
        total_action = np.zeros(self.model.nu, dtype=np.float32)
        total_action[self.maximus_actuator_inds] = actions["maximus"]
        total_action[self.commodus_actuator_inds] = actions["commodus"]

        # -- apply EMA smoothing --
        total_action = (
            self._action_alpha * total_action
            + (1.0 - self._action_alpha) * self._last_action
        )
        # -- save last action --
        self._last_action = total_action
        # -- apply deadband --
        total_action = np.where(
            np.abs(total_action) < self._deadband, 0.0, total_action
        )
        self.do_simulation(total_action, self.frame_skip)

        observation = self._get_obs()
        self._match_time += self.dt
        rewards, terminations, truncations = self._get_rew()
        self._qvel_tm2 = self._qvel_tm1.copy()
        self._qvel_tm1 = self.data.qvel.copy()
        infos = {"maximus": {}, "commodus": {}}

        if self.render_mode == "human":
            self.render()

        return observation, rewards, terminations, truncations, infos

    def _get_rew(self):
        maximus_z = self.data.qpos[2]
        commodus_z = self.data.qpos[11]
        time_up = self._match_time > self._match_length

        # -- termination condition for win/lose --
        maximus_out = maximus_z < -1.0
        commodus_out = commodus_z < -1.0

        # -- win/lose reward --
        active = {"maximus": 0.0, "commodus": 0.0}
        draw = {"maximus": -100.0, "commodus": -100.0}
        max_win = {"maximus": 100.0, "commodus": -100.0}
        com_win = {"maximus": -100.0, "commodus": 100.0}

        if not time_up:
            if not maximus_out and not commodus_out:
                terminations = {"maximus": False, "commodus": False}
                truncations = {"maximus": False, "commodus": False}
                reward = active
            elif not maximus_out and commodus_out:
                terminations = {"maximus": True, "commodus": True}
                truncations = {"maximus": False, "commodus": False}
                reward = max_win
            elif maximus_out and not commodus_out:
                terminations = {"maximus": True, "commodus": True}
                truncations = {"maximus": False, "commodus": False}
                reward = com_win
            else:
                # rare: both out
                terminations = {"maximus": True, "commodus": True}
                truncations = {"maximus": False, "commodus": False}
                reward = draw
        else:
            terminations = {"maximus": False, "commodus": False}
            truncations = {"maximus": True, "commodus": True}
            reward = draw

        if self._train:
            if self._contact_rew_weight > 0.0:
                # -- exploration reward (maximize contact) --
                maximus_contact = self.data.cfrc_ext[1].flatten()[:3]
                commodus_contact = self.data.cfrc_ext[
                    self.data.cfrc_ext.shape[0] // 2 + 1
                ].flatten()[:3]
                reward["maximus"] += self._contact_rew_weight * np.linalg.norm(
                    maximus_contact
                )
                reward["commodus"] += self._contact_rew_weight * np.linalg.norm(
                    commodus_contact
                )

        return reward, terminations, truncations

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        return MujocoEnv.render(self)

    def reset(self, seed=None, options=None):
        self._match_time = 0.0
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        if self._train:
            # -- during training, randomly translate and rotate roombas around the octagon --
            theta1, theta2 = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(2,))
            qpos[0:2] = np.random.uniform(low=-0.75, high=0.75, size=(2,))
            qpos[3:7] = np.array([np.cos(theta1 / 2), 0, 0, np.sin(theta1 / 2)])
            qpos[9:11] = np.random.uniform(low=-0.75, high=0.75, size=(2,))
            qpos[12:16] = np.array([np.cos(theta2 / 2), 0, 0, np.sin(theta2 / 2)])
        self.set_state(qpos, qvel)
        self._qvel_tm1 = self.data.qvel.copy()
        self._qvel_tm2 = self._qvel_tm1.copy()
        self._maximus_pos_history.clear()
        self._commodus_pos_history.clear()
        self._maximus_vel_history.clear()
        self._commodus_vel_history.clear()
        observation = self._get_obs()
        # don't reset deque so you have different filtering inits
        return observation, {"maximus": {}, "commodus": {}}
