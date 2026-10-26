"""C2 验收第 2、3 条：新增观测项 / 修改奖励权重，并真正跑起来

作用
    用一个"子类化任务配置"的方式，不改 Isaac Lab 源码，也不新建外部任务包：
    在内存里继承 FrankaReachEnvCfg，覆盖一个字段，然后用 ManagerBasedRLEnv
    真正构造环境，跑几千步，观察观测维度和奖励的变化。

    做两件 C2 要证明的事：
      1. 新增一个观测项 → 环境真的多了一维观测（不是"我写进了 cfg"而已）
      2. 改奖励权重   → reward 的数值真的变了（这是"改奖励看行为变化"的第一步：
                       先证明数变了，再谈行为变没变）

为什么不用 isaaclab train 做这件事
    `isaaclab train --task <已注册任务>` 只能跑注册过的任务配置。
    C2 要的是"改一个字段看效果"，用子类化 + 直接构造环境，几十秒就能验证，
    不用等完整训练跑完。**这是 C2 和 C3 的本质区别**：
    C2 验证"我改的东西生效了"，C3 才要求"训练到目标成功率"。

为什么必须启动仿真（这次不能像 c2_inspect_managers.py 那样只读）
    观测维度和奖励数值都依赖真实场景（关节数、物理状态），
    配置文件里只有"有哪些项"，没有"值是多少"。所以这里必须 AppLauncher。

运行方式（会启动 Isaac Sim，约 40 秒；显存 < 2 GB）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\c2_obs_and_reward.py

    只跑某一项：
    python ...\\c2_obs_and_reward.py --case obs
    python ...\\c2_obs_and_reward.py --case reward
"""

from __future__ import annotations

import argparse
import sys


def resolve(cfg):
    """把配置里的 PresetCfg（_Preset）全部展开成 default 那一套。

    为什么必须做这一步：Isaac Lab 3.0 把同一个任务的多套配置（default / hold /
    diffik / newton_ik …）用 PresetCfg 包在 cfg 里，避免为每套预设复制一份任务定义。
    **不展开时 `env_cfg.actions.arm_action` 拿到的是一个 Preset 对象而不是动作配置**，
    官方 validate_config() 会误判成"启用了 hold 预设"并抛
    ValueError: The 'hold' preset requires the Franka joint-position action。

    官方 CLI（isaaclab train / play）在构造环境前会自动调用 resolve_presets；
    我们手工构造环境时必须自己调，否则绕过了一整层初始化。
    """
    from isaaclab_tasks.utils.hydra import resolve_presets

    return resolve_presets(cfg)


def build_cases(num_envs: int, steps: int):
    """返回 [(名称, 配置工厂, 说明)]。

    配置工厂是一个返回 env_cfg 的函数：这样每一项 case 都是"干净从默认配置出发"，
    不会因为前一个 case 的修改而串味。
    """
    import isaaclab.envs.mdp as mdp
    import isaaclab_tasks  # noqa: F401
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import (
        FrankaReachEnvCfg,
    )

    def base() -> "FrankaReachEnvCfg":
        cfg = FrankaReachEnvCfg()
        cfg.scene.num_envs = num_envs
        cfg.sim.device = "cuda:0"
        return cfg    # ---------------- case 1：基线（不改任何东西）
    def make_base():
        return base()

    # ---------------- case 2：新增一个观测项（关节力矩）
    @configclass
    class ObsPlusCfg(FrankaReachEnvCfg):
        """"在 policy 观测组里多加一项 joint_effort_rel。"""

        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = num_envs
            self.sim.device = "cuda:0"
            self.observations.policy.joint_effort = ObsTerm(
                func=mdp.joint_effort,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
            )

    def make_obs():
        return ObsPlusCfg()

    # ---------------- case 3：把位置跟踪奖励权重放大 100 倍
    @configclass
    class RewardBigCfg(FrankaReachEnvCfg):
        """位置跟踪奖励权重 -0.2 → -20.0（放大 100 倍）。"""

        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = num_envs
            self.sim.device = "cuda:0"
            self.rewards.end_effector_position_tracking.weight = -20.0

    def make_reward():
        return RewardBigCfg()

    # ---------------- case 4：把位置跟踪奖励权重改成 0（关掉）
    @configclass
    class RewardOffCfg(FrankaReachEnvCfg):
        """位置跟踪奖励权重 -0.2 → 0（等于关掉这一项）。"""

        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = num_envs
            self.sim.device = "cuda:0"
            self.rewards.end_effector_position_tracking.weight = 0.0

    def make_reward_off():
        return RewardOffCfg()

    return [
        ("baseline", make_base, "原始配置，作为对照"),
        ("obs+joint_effort", make_obs, "policy 观测组新增 joint_effort 一项"),
        ("reward x100", make_reward, "位置跟踪权重 -0.2 → -20.0"),
        ("reward off", make_reward_off, "位置跟踪权重 -0.2 → 0.0"),
    ]


def run_case(name: str, factory, note: str, steps: int) -> dict | None:
    import gymnasium as gym
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    print("\n" + "=" * 90)
    print(f"[case] {name} —— {note}")
    print("=" * 90)

    env_cfg = resolve(factory())
    env = None
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
        obs, _ = env.reset()

        # 观测维度
        obs_dim = obs["policy"].shape[-1] if isinstance(obs, dict) else obs.shape[-1]
        n_obs_terms = len([k for k in vars(env_cfg.observations.policy)
                           if not k.startswith("_") and hasattr(
                               getattr(env_cfg.observations.policy, k), "func")])
        print(f"  观测项数: {n_obs_terms}   policy 观测维度: {obs_dim}   num_envs: {env.num_envs}")

        # 奖励项权重
        print("  奖励权重（运行时实际生效的值）:")
        for term_name in env.reward_manager.active_terms:
            term = env.reward_manager.get_term_cfg(term_name)
            print(f"    {term_name:<36} w={term.weight}")

        # 跑若干步，统计各奖励分项与总奖励
        step_dt = env.cfg.sim.dt * env.cfg.decimation
        total_rewards = []
        term_sum = {}
        for i in range(steps):
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.device)
                obs, rew, terminated, truncated, _ = env.step(actions)
                total_rewards.append(float(rew.mean()))
                # 分项奖励：_step_reward 的形状是 (num_envs, num_terms)，
                # 第一维是并行环境、第二维才对应 active_terms。
                # （一开始按 (num_terms, num_envs) 读，导致权重乘 100 倍只算出 21 倍，
                #   还出现"权重设 0 但分项不为 0"的假象——维度顺序反了。）
                sr = env.reward_manager._step_reward
                if sr is not None:
                    for ti, tname in enumerate(env.reward_manager.active_terms):
                        term_sum[tname] = term_sum.get(tname, 0.0) + float(sr[:, ti].mean())
                if i == 0:
                    print(f"  控制频率 {1.0 / step_dt:.1f} Hz，"
                          f"跑 {steps} 步 = 仿真 {steps * step_dt:.2f} s")
                    print(f"  _step_reward 形状={tuple(sr.shape)} "
                          f"(num_envs, num_terms)  已乘权重")

        mean_rew = sum(total_rewards) / len(total_rewards)
        print(f"  总奖励均值（{steps} 步，零动作）: {mean_rew:+.4f}")
        if term_sum:
            print(f"  分项累计（{steps} 步求和）:")
            for k, v in term_sum.items():
                print(f"    {k:<36} {v:+.3f}")

        return dict(name=name, obs_dim=obs_dim, n_terms=n_obs_terms,
                    mean_reward=mean_rew, per_term=term_sum)
    except Exception as exc:
        print(f"  [失败] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def summarize(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("【结果对比】")
    print("=" * 90)
    base = next((r for r in results if r and r["name"] == "baseline"), None)

    print(f"{'case':<20} {'观测维度':>8} {'观测项数':>8} {'总奖励均值':>12}")
    print("-" * 90)
    for r in results:
        if not r:
            continue
        print(f"{r['name']:<20} {r['obs_dim']:>8} {r['n_terms']:>8} {r['mean_reward']:>+12.4f}")

    obs_case = next((r for r in results if r and r["name"] == "obs+joint_effort"), None)
    if base and obs_case:
        delta = obs_case["obs_dim"] - base["obs_dim"]
        print(f"\n验收第 2 条：新增观测项后维度 {base['obs_dim']} → {obs_case['obs_dim']}"
              f"（{delta:+d}）")
        print("  " + ("✅ 生效：确实多了一维（7 个 panda 关节的力矩）"
                     if delta == 7 else f"⚠ 变化 {delta}，预期 +7，需检查新增项覆盖的关节数"))

    if base:
        for label in ("reward x100", "reward off"):
            r = next((x for x in results if x and x["name"] == label), None)
            if not r:
                continue
            key = "end_effector_position_tracking"
            b = base["per_term"].get(key, 0.0)
            c = r["per_term"].get(key, 0.0)
            print(f"\n验收第 3 条（{label}）：{key} 分项累计 {b:+.3f} → {c:+.3f}")
            if abs(b) > 1e-9:
                print(f"  比值 = {c / b:.2f}x（预期 x100 时约 100 倍，off 时为 0）")


def main() -> int:
    ap = argparse.ArgumentParser(description="C2: 观测项与奖励项的真实验证")
    ap.add_argument("--case", choices=["all", "obs", "reward"], default="all")
    ap.add_argument("--num_envs", type=int, default=64, help="只验配置生效，64 就够")
    ap.add_argument("--steps", type=int, default=120, help="跑多少控制步（120 步 = 4 秒仿真）")
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    # 必须最先启动 App（在 import torch/gym 之前由 launcher 接管）
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app

    cases = build_cases(args.num_envs, args.steps)
    if args.case == "obs":
        cases = [c for c in cases if c[0] in ("baseline", "obs+joint_effort")]
    elif args.case == "reward":
        cases = [c for c in cases if c[0] in ("baseline", "reward x100", "reward off")]

    results = []
    for name, factory, note in cases:
        results.append(run_case(name, factory, note, args.steps))

    summarize(results)

    simulation_app.close()
    print("\n[完成] 把上面的对比贴进总计划 C2 节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
