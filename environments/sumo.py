__credits__ = ["Gianluca"]

from typing import Dict, Tuple, Union

import os
import numpy as np

import mujoco

from pettingzoo import ParallelEnv

from gymnasium import utils
from .mujoco_env import MujocoEnv
from gymnasium.spaces import Box

CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_XML_PATH = os.path.join(CURRENT_DIR, "assets", "sumo.xml")
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 0.2)),
    "elevation": -45.0,
}

class Sumo(ParallelEnv, MujocoEnv, utils.EzPickle):
    metadata = {
        "name": "sumo_cpu",
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self, 
        xml_file: str = DEFAULT_XML_PATH,
        frame_skip: int = 20, # sim @ 200hz, actions @ 10hz
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        match_length: float = 60.0, # 1-minute matches
        reset_noise_scale: float = 1e-2,
        contact_rew_weight: float = 1e-1,
        train: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            match_length,
            reset_noise_scale,
            train,
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
            "maximus": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "commodus": Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        }

    @property
    def possible_agents(self):
        return ["maximus", "commodus"]

    @property
    def agents(self):
        return ["maximus", "commodus"]

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        # -- x, y position & velocity --
        maximus_xy_pos = qpos[0:2]
        commodus_xy_pos = qpos[9:11]
        maximus_xy_vel = qvel[0:2]
        commodus_xy_vel = qvel[8:10]

        # -- quaternions (w,x,y,z) --
        maximus_quat = qpos[3:7]
        commodus_quat = qpos[12:16]

        # -- heading --
        def yaw_sincos(q):
            w,x,y,z = q
            return 2*(w*z + x*y), 1 - 2*(y*y + z*z)

        maximus_sin, maximus_cos = yaw_sincos(maximus_quat)
        commodus_sin, commodus_cos = yaw_sincos(commodus_quat)

        # -- velocity of left and right wheels --
        maximus_wdotL, maximus_wdotR = qvel[6], qvel[7]
        commodus_wdotL, commodus_wdotR = qvel[14], qvel[15]

        # -- distance to center --
        maximus_dist = np.linalg.norm(maximus_xy_pos)
        commodus_dist = np.linalg.norm(commodus_xy_pos)

        # -- torques --
        maximus_torqueL, maximus_torqueR = self.data.qfrc_actuator[0], self.data.qfrc_actuator[1]
        commodus_torqueL, commodus_torqueR = self.data.qfrc_actuator[2], self.data.qfrc_actuator[3]

        # -- maximus to commodus relative --
        rel_pos  = maximus_xy_pos - commodus_xy_pos
        rel_vel  = maximus_xy_vel - commodus_xy_vel 

        # -- external forces --
        maximus_contact = self.data.cfrc_ext[1].flatten()
        commodus_contact = self.data.cfrc_ext[self.data.cfrc_ext.shape[0] // 2 + 1].flatten()

        # TODO: add exploration reward that is about velocity, perhaps check if rel velocity is negative and then reward based on world vel

        maximus_obs = np.concatenate([np.array([maximus_dist, maximus_sin, maximus_cos, maximus_wdotL, maximus_wdotR, maximus_torqueL, maximus_torqueR]), maximus_contact, rel_pos, rel_vel]) 
        commodus_obs = np.concatenate([np.array([commodus_dist, commodus_sin, commodus_cos, commodus_wdotL, commodus_wdotR, commodus_torqueL, commodus_torqueR]), commodus_contact, -rel_pos, -rel_vel])
        return {
            "maximus": maximus_obs,
            "commodus": commodus_obs,
        } 

    def step(self, actions):
        total_action = np.zeros(self.model.nu, dtype=np.float32)
        total_action[self.maximus_actuator_inds] = actions["maximus"]
        total_action[self.commodus_actuator_inds] = actions["commodus"]

        self.do_simulation(total_action, self.frame_skip)

        observation = self._get_obs()
        self._match_time += self.dt
        rewards, terminations, truncations = self._get_rew()
        infos = {"maximus": {}, "commodus": {}}

        if self.render_mode == "human":
            self.render()

        return observation, rewards, terminations, truncations, infos

    def _get_rew(self):
        maximus_z = self.data.qpos[2]
        commodus_z = self.data.qpos[11]
        time_up = self._match_time > self._match_length

        maximus_out = maximus_z < -1.0
        commodus_out = commodus_z < -1.0

        active = {"maximus": 0., "commodus": 0.}
        draw = {"maximus": -1000., "commodus": -1000.}
        max_win = {"maximus": 1000., "commodus": -1000.}
        com_win = {"maximus": -1000., "commodus": 1000.}

        # TODO: add exploration reward that is about contact (make weight = 1e-2)
        # print(np.linalg.norm(maximus_contact))
        # print(np.linalg.norm(commodus_contact))

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

        # exploration reward
        if self._train and self._contact_rew_weight > 1e-4:
            maximus_contact = self.data.cfrc_ext[1].flatten()
            commodus_contact = self.data.cfrc_ext[self.data.cfrc_ext.shape[0] // 2 + 1].flatten()
            reward["maximus"] += self._contact_rew_weight * np.linalg.norm(maximus_contact) 
            reward["commodus"] += self._contact_rew_weight * np.linalg.norm(commodus_contact) 

        return reward, terminations, truncations


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self, mode="human"):
        return MujocoEnv.render(self)

    def reset(self, seed=None, options=None):
        self._match_time = 0.0
        self._contact_rew_weight *= 0.95
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        if self._train:
            theta1, theta2 = np.random.uniform(low=0., high=2.*np.pi, size=(2,))
            qpos[0:2] = np.random.uniform(low=-1.5, high=1.5, size=(2,))
            qpos[3:7] = np.array([np.cos(theta1/2), 0, 0, np.sin(theta1/2)])
            qpos[9:11] = np.random.uniform(low=-1.5, high=1.5, size=(2,))
            qpos[12:16] = np.array([np.cos(theta2/2), 0, 0, np.sin(theta2/2)])
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation, {"maximus": {}, "commodus": {}}
        
