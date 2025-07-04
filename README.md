# SFsL: PPO and DDPG Comparison

This repository contains a simplified simulation of the SFsL scenario used in our paper, including PPO and DDPG training under a multi-client reliability-aware RL environment.

### 📄 Paper Reference

- **State/Action Design** → Sec. III-A, Eq. (55)–(57)  
- **Reliability Reward** → Eq. (43)  
- **Lyapunov Queues (light version)** → Eq. (46)–(50)

### 🔧 How to Run

```bash
pip install stable-baselines3 torch gym matplotlib
python ppo.py
