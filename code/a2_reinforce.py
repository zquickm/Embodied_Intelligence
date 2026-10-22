"""A2 第一部分：手写 REINFORCE + baseline 解 CartPole-v1

作用
    不依赖任何 RL 库，用 PyTorch 从零实现策略梯度算法，在 CartPole-v1 上训练到
    目标回报（近 100 回合平均 >= 475）。这是 A2 第 3 条验收标准
    （"能手推策略梯度公式"）的代码落地：每一行都能对应到公式里的一项。

为什么会有这个文件
    计划模块 A 里没有单独讲策略梯度的阶段，但 A3 的 PPO 是策略梯度的改进版。
    如果不先把"最朴素的那一版"手写一遍，学 PPO 时既有 clip 又有 GAE，
    会分不清哪些是策略梯度的本质、哪些是工程技巧。
    REINFORCE 只有 60 行核心代码，是理解后面一切的基准。

对整个项目的作用
    - A3 PPO：本文件的策略网络 + 优势估计就是 PPO 的骨架，PPO 额外加的是
      重要性采样（一条轨迹更新多次）和 clip（限制更新幅度）
    - A4 SAC：换成"学 Q 函数 + actor 直接输出动作"，但策略梯度的核心
      ∇log π · A 完全一样
    - D3 Dreamer：actor 的更新用的也是这套策略梯度（在世界模型里想象出的
      轨迹上算），理解本文件才能理解"在想象中训练"改了什么
    - on-policy/off-policy 对比：本文件是 on-policy 的原型，配 a2_dqn.py 看

运行方式（不要在仓库目录跑）
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\a2_reinforce.py --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym


class Policy(nn.Module):
    """Actor-Critic 共享主干：一个 trunk，两个头（策略头 + 价值头）。

    为什么共享主干：
        CartPole 的观测只有 4 维，两个独立网络会浪费容量且训练更慢。
        共享特征提取层是 A3 PPO、A4 SAC 的标准做法。

    为什么价值头存在（明明 REINFORCE 不需要价值函数）：
        价值函数作为基线 b(s) 降低梯度方差。没有它，同一状态下不同轨迹的
        回报差异会直接进梯度，收敛极慢甚至不收敛。
        注意：基线只降方差、不引入偏差（推导见 阶段A2 文档 3.4 节）。
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)   # 输出每个动作的 logit
        self.v = nn.Linear(hidden, 1)          # 输出状态价值的一个标量

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)


def discounted_returns(rewards, gamma, device):
    """从后往前累加，算每步的折扣回报 G_t = r_{t+1} + γ r_{t+2} + ...

    为什么必须倒着算：
        G_t 依赖 G_{t+1}（G_t = r_{t+1} + γ G_{t+1}），倒序遍历是唯一的
        一趟 O(T) 算法。正着算每步都要重新扫一遍后面，O(T^2)。
    """
    out, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=device)


def collect_episode(env, model, device, seed):
    """用当前策略跑一整条轨迹，返回 (loss 需要的东西, 该集总回报)。

    为什么必须跑完整条轨迹：
        REINFORCE 需要 G_t（未来的折扣奖励总和），不跑完根本算不出。
        这是 on-policy 算法的固有代价——A3 的 PPO 也跑完整轨迹，
        但它用重要性采样让一条轨迹能更新多次。
    """
    obs, _ = env.reset(seed=seed)
    logps, values, rewards, entropies = [], [], [], []
    done = False
    while not done:
        s = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits, v = model(s)

        # 动作采样：这一步就是"策略"。Categorical 把 logits 变成可采样的分布，
        # 并记录 log π(a|s)——策略梯度公式里 ∇log π 的那一项。
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()

        logps.append(dist.log_prob(a))
        entropies.append(dist.entropy())   # 熵越大策略越随机；用它鼓励探索
        values.append(v)

        obs, r, terminated, truncated, _ = env.step(a.item())
        rewards.append(r)
        done = terminated or truncated

    return logps, values, rewards, entropies, float(sum(rewards))


def reinforce_update(model, optimizer, logps, values, rewards, entropies, args, device):
    """执行一次策略梯度更新。

    核心是这一行：loss = -(log π(a|s) * A).mean()
        - 为什么有负号：优化器做的是"最小化"，而我们要"最大化"回报，
          所以取负变成最小化负回报。
        - 为什么要乘 A（优势）：A > 0 的动作（比基线好）应被提高概率，
          A < 0 的应被降低。乘上 A 就自动完成这个方向选择。
    """
    # (1) 算每步回报 G_t，并标准化
    returns = discounted_returns(rewards, args.gamma, device)
    # 标准化回报：让梯度尺度与任务奖励量级无关。
    # 不做这一步，CartPole 的奖励是每步 +1，回报能到 500，
    # 梯度尺度会极大，必须靠很小的学习率才能稳。标准化后 lr 可以直接用 1e-3。
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)

    logps = torch.stack(logps)
    values = torch.stack(values)

    # (2) 优势 A = G_t - V(s_t)。V 是基线，用 detach 切断梯度：
    #     基线只做"常数偏移"，不能让策略损失去优化价值网络（否则目标不一致）。
    advantage = (returns_norm - values).detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # (3) 三个损失项
    policy_loss = -(logps * advantage).mean()          # 策略梯度，带负号表示最大化
    value_loss = F.mse_loss(values, returns_norm)      # 基线回归到实际回报
    entropy = torch.stack(entropies).mean()            # 熵奖励，鼓励探索

    # 为什么 value_loss 要乘 0.5：这个系数是经验值，平衡两个损失的梯度尺度。
    # 为什么减 entropy：最大化熵 = 最小化负熵，与 policy_loss 方向一致。
    loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # 梯度裁剪：策略梯度方差大，偶发的超大梯度会让参数一步走飞，
    # 之后再也回不来（表现为回报突然掉到 10 以下）。裁剪是最便宜的保险。
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss), float(value_loss), float(entropy)


def main():
    p = argparse.ArgumentParser(description="A2: hand-written REINFORCE on CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1200, help="训练总集数（所有种子用同一预算）")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-3)
    # 熵系数默认 0：实测 0.01 时 CartPole 只能到 375 分左右，策略被熵奖励拖着
    # 保持过度随机，永远学不到"每步都存活"的确定性最优行为。设 0 后到 500。
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--stop_on_solve", action="store_true",
                   help="达 target+margin 即停，节省多种子实验时间")
    p.add_argument("--stop_margin", type=float, default=15.0,
                   help="提前停止的余量：avg100 >= target+margin 才停")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"],
                   help="默认 cpu：实测 CartPole 上 CPU 比 GPU 快约 2 倍")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--target", type=float, default=475.0, help="CartPole-v1 的 reward_threshold")
    p.add_argument("--out_dir", default=r"D:\isaac\work")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device={device} seed={args.seed} episodes={args.episodes}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = Policy(obs_dim, act_dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    returns_hist, solved_at = [], None
    best_avg, ckpt = -1e9, out / f"a2_cartpole_reinforce_seed{args.seed}.pt"
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        # 每个种子用固定的 reset 序列：不同种子看到不同的初始状态序列，保证可复现
        logps, values, rewards, entropies, ep_ret = collect_episode(
            env, model, device, seed=args.seed * 1_000_000 + ep
        )
        reinforce_update(model, optimizer, logps, values, rewards, entropies, args, device)
        returns_hist.append(ep_ret)

        # 每集都检查"近 100 集平均"，一旦达标就记录第一次达标的集数
        if len(returns_hist) >= 100:
            avg = float(np.mean(returns_hist[-100:]))
            if avg > best_avg:
                best_avg = avg
                torch.save({"episode": ep, "model": model.state_dict(), "avg100": avg}, ckpt)
            if solved_at is None and avg >= args.target:
                solved_at = ep
                print(f"solved at ep {ep} (avg100={avg:.1f}, {time.time()-t0:.0f}s)")
        if args.stop_on_solve and solved_at is not None and best_avg >= args.target + args.stop_margin:
            print(f"early stop: solved at ep {solved_at}, budget {args.episodes}")
            break
        if ep % 100 == 0:
            recent = float(np.mean(returns_hist[-100:]))
            print(f"ep {ep:4d} avg100={recent:6.1f} last={ep_ret:5.1f} {time.time()-t0:4.0f}s")

    # 用"近 100 集平均"作为最终成绩，而不是最后一集
    # （最后一集可能是运气好或运气差，单集数字没有统计意义）
    final_avg = float(np.mean(returns_hist[-100:]))
    print(f"FINAL seed={args.seed} avg100={final_avg:.1f} solved_at={solved_at} "
          f"best_avg={best_avg:.1f} wall={time.time()-t0:.0f}s")

    # 曲线：横轴集数，纵轴回报，叠加 100 集滑动平均
    plt.figure(figsize=(8, 4.5))
    plt.plot(returns_hist, alpha=0.25, lw=0.7, label="episode return")
    if len(returns_hist) >= 100:
        ma = np.convolve(returns_hist, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(returns_hist)), ma, lw=1.8, label="100-episode mean")
    plt.axhline(args.target, ls="--", c="gray", label=f"target {args.target:.0f}")
    if solved_at:
        plt.axvline(solved_at, ls=":", c="green", label=f"solved @ep{solved_at}")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = out / f"a2_cartpole_seed{args.seed}.png"
    plt.savefig(fig, dpi=130)
    plt.close()

    print(json.dumps({
        "algo": "REINFORCE+baseline", "env": "CartPole-v1", "seed": args.seed,
        "episodes": args.episodes, "gamma": args.gamma, "lr": args.lr,
        "ent_coef": args.ent_coef, "final_avg100": round(final_avg, 1),
        "best_avg100": round(best_avg, 1), "solved_at": solved_at,
        "solved": solved_at is not None, "target": args.target,
    }))


if __name__ == "__main__":
    main()
