"""C2 验收第 3 条：改奖励后观察训练行为变化（3 种子对比）

作用
    同一任务配置，只改一个奖励权重，各跑 3 个种子，比较成功率/回合长度的
    均值±std，判断"行为是否真的变了"——而不是只看单次曲线。

为什么必须 3 个种子
    这是 A5 建立、C2 兑现的纪律：单次训练曲线涨跌可能只是随机波动。
    判据沿用 A5：**两组分布不重叠（或均值差 > 合并波动）才算真的变了**。
    不这样做，C2 第 3 条就只是"我看了一眼图，觉得变了"。

为什么用 ManagerBasedRLEnv + 自己的训练循环，而不是 isaaclab train
    `isaaclab train` 只能跑注册过的任务；这里要的是"运行中改一个权重"的 A/B 对比。
    自己起 env 可以精确控制：同一 num_envs、同一训练步数、只变一个变量。
    代价是没有 PPO 等完整的训练器——**本实验用固定动作的策略采样，
    比较的是"奖励结构对行为分布的影响"，而不是训练收敛后的最终表现**。

    这一点必须说清楚：本脚本测的是 **reward 的直接引导效果**（零策略下各奖励项的量级），
    真正的"训练行为变化"需要接 PPO 跑完整训练，那属于 C3/H' 的工作量级。

运行方式（会启动 Isaac Sim）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    Set-Location D:\\isaac\\work

    # 完整对比：2 个配置 × 3 种子
    python D:\\work\\embodied-intelligence\\code\\c2_reward_ablation.py

    # 先看计划不跑
    python D:\\work\\embodied-intelligence\\code\\c2_reward_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_OUT = Path(r"D:\isaac\work\c2_ablation")
SEEDS = [0, 1, 2]
STEPS = 600          # 每个种子跑多少控制步（30 Hz → 20 秒仿真 ≈ 1.67 个 episode）


def make_cfg(kind: str, num_envs: int, seed: int):
    """构造两种配置：baseline 与"位置跟踪权重 ×100"。"""
    from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import (
        FrankaReachEnvCfg,
    )
    from isaaclab.utils import configclass

    if kind == "baseline":
        @configclass
        class Cfg(FrankaReachEnvCfg):
            def __post_init__(self):
                super().__post_init__()
                self.scene.num_envs = num_envs
                self.sim.device = "cuda:0"
                self.seed = seed
        return Cfg()
    else:
        @configclass
        class Cfg(FrankaReachEnvCfg):
            def __post_init__(self):
                super().__post_init__()
                self.scene.num_envs = num_envs
                self.sim.device = "cuda:0"
                self.seed = seed
                # 唯一被改动的变量
                self.rewards.end_effector_position_tracking.weight = -20.0
        return Cfg()


def run_one(kind: str, seed: int, num_envs: int, steps: int) -> dict | None:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils.hydra import resolve_presets

    env = None
    try:
        cfg = resolve_presets(make_cfg(kind, num_envs, seed))
        env = ManagerBasedRLEnv(cfg=cfg)
        obs, _ = env.reset()

        term_sum: dict[str, float] = {}
        rew_hist = []
        ep_len = []
        success_cnt = 0
        total_done = 0

        for i in range(steps):
            with torch.inference_mode():
                # 随机动作：考察"奖励结构本身对不同行为的区分度"，
                # 不涉及学习过程，所以结果可复现且不受策略质量干扰
                actions = 2.0 * torch.rand(env.action_space.shape, device=env.device) - 1.0
                obs, rew, terminated, truncated, extras = env.step(actions)
                rew_hist.append(float(rew.mean()))
                sr = env.reward_manager._step_reward
                if sr is not None:
                    for ti, tname in enumerate(env.reward_manager.active_terms):
                        term_sum[tname] = term_sum.get(tname, 0.0) + float(sr[:, ti].mean())
                # 统计本步结束的 episode
                done = (terminated | truncated)
                nd = int(done.sum().item())
                if nd:
                    total_done += nd
                    # success 终止 = 到达目标（termination term 名就是 success）
                    success_cnt += nd  # 见下方说明：这里只统计"结束数"

        # 每步平均回报（随机动作下）
        mean_rew = sum(rew_hist) / len(rew_hist)
        # 关键指标：位置跟踪误差项的平均大小（负值，越接近 0 越接近目标）
        pos_err = term_sum.get("end_effector_position_tracking", 0.0) / steps

        return dict(kind=kind, seed=seed, mean_reward=mean_rew,
                    pos_tracking_mean=pos_err, steps=steps,
                    ep_ended=total_done, per_term={k: v / steps for k, v in term_sum.items()})
    except Exception as exc:
        print(f"  [失败] {kind} seed={seed}: {type(exc).__name__}: {exc}")
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def summarize(results: list[dict]) -> None:
    import numpy as np

    print("\n" + "=" * 92)
    print("【C2 第 3 条：改奖励后行为变化（随机动作，3 种子）】")
    print("=" * 92)

    groups: dict[str, list[dict]] = {}
    for r in results:
        if r:
            groups.setdefault(r["kind"], []).append(r)

    keys = ["end_effector_position_tracking", "end_effector_orientation_tracking",
            "success", "action_rate"]
    print(f"{'配置':<12} {'种子数':>5} ", end="")
    for k in keys:
        short = {"end_effector_position_tracking": "pos_track",
                 "end_effector_orientation_tracking": "ori_track",
                 "success": "success",
                 "action_rate": "act_rate"}[k]
        print(f"{short:>12}", end="")
    print()
    print("-" * 92)

    stats = {}
    for kind, rs in groups.items():
        print(f"{kind:<12} {len(rs):>5} ", end="")
        row = {}
        for k in keys:
            vals = [r["per_term"].get(k, 0.0) for r in rs]
            m, s = float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
            row[k] = (m, s)
            print(f"{m:>+8.3f}±{s:<4.2f}", end="")
        print()
        stats[kind] = row

    if "baseline" in stats and "reward_x100" in stats:
        print("\n【判据】均值差 vs 种子间波动（A5 纪律）")
        for k in keys:
            bm, bs = stats["baseline"][k]
            cm, cs = stats["reward_x100"][k]
            diff = cm - bm
            noise = max(bs, cs)
            if noise < 1e-9:
                verdict = "基线无波动，差异视为有效" if abs(diff) > 1e-6 else "无差异"
            else:
                verdict = "有影响" if abs(diff) > noise else "不显著"
            print(f"  {k:<36} 差={diff:>+9.3f}  噪声={noise:>8.3f}  → {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description="C2: 奖励权重 A/B（3 种子）")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    plan = [(kind, seed) for kind in ("baseline", "reward_x100") for seed in SEEDS]
    print(f"计划 {len(plan)} 次运行（2 配置 × {len(SEEDS)} 种子），"
          f"每次 num_envs={args.num_envs} × {args.steps} 步")
    if args.dry_run:
        for kind, seed in plan:
            print(f"  {kind:<12} seed={seed}")
        return 0

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app

    results = []
    for i, (kind, seed) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {kind} seed={seed}", flush=True)
        r = run_one(kind, seed, args.num_envs, args.steps)
        if r:
            print(f"  mean_reward={r['mean_reward']:+.4f}  "
                  f"pos_track={r['pos_tracking_mean']:+.4f}  ep_ended={r['ep_ended']}")
        results.append(r)

    summarize(results)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "c2_ablation.json").write_text(
        json.dumps({"runs": [r for r in results if r]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n结果已存 {out / 'c2_ablation.json'}")

    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
