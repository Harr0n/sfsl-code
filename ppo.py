"""
SFsL  |  PPO & DDPG
Paper section reference:
  – State/action design   →  Sec. III-A, Eq.(55)–(57)
  – Reliability reward    →  Eq.(43)
  – Lyapunov queues       →  Eq.(46)–(50)
"""

import numpy as np
import gym
from gym import spaces
import torch
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import os, random

# ----------------------- Environment ----------------------- #
NUM_CLI = 3                 # number of physical-twin clients
T_THRESH = 8e3              # time budget per slot   (arbitrary units)
E_THRESH = 5e3              # energy budget per slot (arbitrary units)


class SFsFLPPOEnv(gym.Env):

    def __init__(self, seed: int = 0):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 10.0, 20.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 20.0, 30.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5,  dtype=np.float32),
        )

        self.seed(seed)

    # ---------- helpers ---------- #
    def _normalize_state(self, delta_D, loss, rb, fk, fe):
        return np.array([
            delta_D / 5_000.0,            # ΔD_i ∈ [0,5000]
            loss    / 5.0,                # Loss  ∈ [0,5]
            rb      / 10.0,               # RB    ∈ [0,10]
            (fk - 2.0) / 1.5,             # f_k   ∈ [2,3.5]
            (fe - 10.) / 1.0              # f_e   = 10
        ], dtype=np.float32)

    def _system_state_vector(self):
        return self._normalize_state(
            self.data_size.mean(),
            self.loss_vec.mean(),
            self.rb,
            self.cpu_pt.mean(),
            self.cpu_edge,
        )

    # ---------- gym core ---------- #
    def seed(self, seed=None):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def reset(self):
        self.data_size = np.random.randint(2_000, 5_000, size=NUM_CLI)
        self.loss_vec  = np.random.uniform(1.0, 3.0,  size=NUM_CLI)
        self.cpu_pt    = np.random.uniform(2.0, 3.5,  size=NUM_CLI)
        self.rb        = np.random.randint(1, 10)
        self.cpu_edge  = 10.0
        self.Q_T, self.Q_E = 0.0, 0.0
        return self._system_state_vector()

    # ----- reward (Eq.(43) + Lyapunov) ----- #
    def _compute_reward(self, lr_k, th_dc, th_adj, rho_k, rho_e):
        time_vec   = self.data_size / (self.cpu_pt * lr_k + 1e-6)
        energy_vec = th_dc * rho_k + th_adj * rho_e

        acc_vec    = np.maximum(0., 1. - self.loss_vec * (1 - th_adj))
        acc_glb    = acc_vec.sum()
        cost_glb   = (time_vec + energy_vec).sum() + 1e-6
        reliability= acc_glb / cost_glb

        self.Q_T = max(0., self.Q_T + cost_glb - T_THRESH)
        self.Q_E = max(0., self.Q_E + energy_vec.sum() - E_THRESH)
        penalty  = 0.01 * (self.Q_T + self.Q_E)

        return reliability - penalty

    def step(self, action):
        lr_k, th_dc, th_adj, rho_k, rho_e = action

        reward = self._compute_reward(lr_k, th_dc, th_adj, rho_k, rho_e)

        self.data_size = np.maximum(0, self.data_size - (1000 * lr_k).astype(int))
        self.loss_vec *= (0.95 + 0.05 * np.random.rand(NUM_CLI))
        done = False
        return self._system_state_vector(), reward, done, {}


# ----------------------- training utils ----------------------- #
def train_agent(algo="ppo", total_steps=100_000, seed=0):
    env = make_vec_env(lambda: SFsFLPPOEnv(seed), n_envs=1)

    if algo == "ppo":
        model = PPO(
            "MlpPolicy", env,
            gamma=0.95, gae_lambda=0.95, clip_range=0.2,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            policy_kwargs=dict(net_arch=[64,64]), verbose=0, seed=seed
        )
    else:  # ddpg
        n_act = env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1*np.ones(n_act))
        model = DDPG(
            "MlpPolicy", env, action_noise=noise,
            learning_rate=3e-4, verbose=0, seed=seed
        )

    model.learn(total_timesteps=total_steps)
    return model, env


def rollout_rewards(model, env, steps=200):
    obs = env.reset()
    rs  = []
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, _, _ = env.step(action)
        rs.append(r.item())
    return rs


# ----------------------- main run ----------------------- #
def main():
    # --- PPO --- #
    ppo_model, ppo_env = train_agent("ppo")
    r_ppo = rollout_rewards(ppo_model, ppo_env)

    # --- DDPG baseline --- #
    ddpg_model, ddpg_env = train_agent("ddpg")
    r_ddpg = rollout_rewards(ddpg_model, ddpg_env)

    # --- plot --- #
    plot_results(r_ppo, r_ddpg)

def plot_results(r_ppo, r_ddpg):
    plt.figure(figsize=(6, 4))
    plt.plot(r_ppo,  label="PPO")
    plt.plot(r_ddpg, label="DDPG")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Reliability")
    plt.title("PPO vs DDPG on SFsFL (3 clients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_vs_ddpg.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
