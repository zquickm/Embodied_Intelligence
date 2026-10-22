"""A2 第二部分：用 SB3 DQN 解 LunarLander-v3

作用
    用 Stable-Baselines3 的 DQN 在 LunarLander-v3 上训练到目标回报（>= 200）。
    DQN 是 off-policy 算法的原型：经验回放 + 目标网络。

为什么会有这个文件
    A2 第 5 条验收标准要求解释 on-policy 与 off-policy 的区别并举算法例子。
    只做一个 on-policy 算法（REINFORCE）无法对比，而模块 A 原本没有 DQN 阶段。
    本文件补上这个缺口，也是 A4 SAC 的直系祖先（SAC = DQN 思想的连续动作版）。

    另外 LunarLander 比 CartPole 难得多（8 维观测、4 个动作、需精确控制着陆姿态），
    手写 REINFORCE 很难稳定达标，用它来认识"什么任务该换更强的算法"很合适。

对整个项目的作用
    - A4 SAC：SAC 的双 Q 网络、目标网络软更新、经验回放全部来自 DQN
    - B 模块连续控制：理解"为什么连续动作不能直接用 DQN"（要对动作取 max，
      连续空间无法枚举），才会明白为什么要 actor-critic
    - D 阶段：世界模型的训练数据也来自这种"与环境交互存 buffer"的范式

运行方式（不要在仓库目录跑）
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\a2_dqn.py --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


class EvalAndSaveBest(BaseCallback):
    """训练过程中定期评估，只保存"评估表现最好"的那个模型。

    为什么要这个 callback（而不是直接用训练结束时的模型）：
        DQN 训练不稳定，表现会先升后降。实测本项目：训练中 ep_rew_mean
        最好到 242，但训练结束的模型评估只有 174.6。
        用最后一个模型等于主动选了个退化的版本。
        "保存验证集上最好的 checkpoint" 和 A1 里按最优 val_acc 存模型是
        同一个原则 —— 评估驱动的模型选择。

    为什么用独立的 eval_env：
        评估时不能污染训练环境的探索状态；评估用 deterministic 动作，
        也不能让评估数据进 replay buffer。
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, save_path: Path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean = -np.inf
        self.history = []      # [(timesteps, mean, std)]

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        mean, std = evaluate_policy(
            self.model, self.eval_env,
            n_eval_episodes=self.n_eval_episodes, deterministic=True,
        )
        self.history.append((self.num_timesteps, float(mean), float(std)))
        if mean > self.best_mean:
            self.best_mean = float(mean)
            self.model.save(str(self.save_path))
            print(f"  [eval] step {self.num_timesteps:7d} mean {mean:7.1f} +- {std:5.1f}  <- best")
        else:
            print(f"  [eval] step {self.num_timesteps:7d} mean {mean:7.1f} +- {std:5.1f}")
        return True


def main():
    p = argparse.ArgumentParser(description="A2: SB3 DQN on LunarLander-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--n_eval", type=int, default=15)
    p.add_argument("--target", type=float, default=200.0, help="LunarLander-v3 的 reward_threshold")
    p.add_argument("--device", default="cpu",
                   help="默认 cpu：这个小网络在 CPU 上更快（GPU kernel 启动开销占主导）")
    p.add_argument("--out_dir", default=r"D:\isaac\work")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    train_env = Monitor(gym.make("LunarLander-v3"))
    eval_env = Monitor(gym.make("LunarLander-v3"))

    # 这些超参是实测出来的，不是抄来的：
    #   300k 步 + lr 5e-4 -> 评估 190.0
    #   650k 步 + lr 5e-4 -> 评估 174.6（后期退化）
    #   700k 步 + lr 1e-3 -> 评估 119.8（lr 太大，不稳定）
    # 结论：lr 5e-4 合适，但必须靠"保存最优 checkpoint"而不是加步数。
    model = DQN(
        "MlpPolicy", train_env, seed=args.seed, verbose=0,
        buffer_size=100_000,        # 经验回放池容量：off-policy 的核心组件
        learning_starts=1_000,      # 先随机探索再开始学习，避免用极少数据拟合
        batch_size=128,             # 每次从 buffer 随机抽的样本数
        gamma=0.99,
        target_update_interval=500, # 目标网络同步间隔（每 500 步硬同步一次）
        train_freq=4,               # 每 4 步环境交互做一次梯度更新
        exploration_fraction=0.2,   # 前 20% 时间 ε 从 1.0 线性衰减到 final_eps
        learning_rate=5e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=args.device,
    )

    best_path = out / f"a2_lunarlander_dqn_seed{args.seed}"
    cb = EvalAndSaveBest(eval_env, args.eval_freq, args.n_eval, best_path)

    print(f"device={args.device} seed={args.seed} timesteps={args.timesteps}")
    model.learn(total_timesteps=args.timesteps, callback=cb)

    # 加载"最好的那个"模型再单独评估一次，作为最终成绩
    best_model = DQN.load(str(best_path) + ".zip", env=eval_env)
    final_mean, final_std = evaluate_policy(
        best_model, eval_env, n_eval_episodes=30, deterministic=True
    )
    print(f"FINAL seed={args.seed} best_eval_mean={cb.best_mean:.1f} "
          f"final30={final_mean:.1f} +- {final_std:.1f} wall={time.time()-t0:.0f}s")

    # 评估曲线：横轴训练步数，纵轴 15 集平均回报
    if cb.history:
        steps = [h[0] for h in cb.history]
        means = [h[1] for h in cb.history]
        stds = [h[2] for h in cb.history]
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, means, "o-", ms=3, label="eval mean")
        plt.fill_between(steps,
                         np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds),
                         alpha=0.2, label="+-1 std")
        plt.axhline(args.target, ls="--", c="gray", label=f"target {args.target:.0f}")
        best_i = int(np.argmax(means))
        plt.axvline(steps[best_i], ls=":", c="green",
                    label=f"best @{steps[best_i]//1000}k ({means[best_i]:.0f})")
        plt.xlabel("timesteps")
        plt.ylabel("eval return")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig = out / f"a2_lunarlander_seed{args.seed}.png"
        plt.savefig(fig, dpi=130)
        plt.close()
        print(f"saved: {fig.name}")

    print(json.dumps({
        "algo": "DQN", "env": "LunarLander-v3", "seed": args.seed,
        "timesteps": args.timesteps, "lr": 5e-4, "gamma": 0.99,
        "best_eval_mean": round(cb.best_mean, 1),
        "final_eval_mean": round(float(final_mean), 1),
        "final_eval_std": round(float(final_std), 1),
        "solved": float(final_mean) >= args.target, "target": args.target,
    }))


if __name__ == "__main__":
    main()
