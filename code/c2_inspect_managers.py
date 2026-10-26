"""C2 实验：只读地"解剖" Isaac-Reach-Franka 的 7 个管理器

作用
    不启动仿真、不训练，只 import 任务配置并把它打印成一张结构表：
    这个任务用到哪几个管理器、每个管理器下有几项、每项的类名/权重/函数。

为什么这是 C2 的第一步
    C2 的验收第 1 条是"能画出 7 个管理器的分工"。打印比读源码快，
    而且能顺手拿到 C1 实际训过的那个任务（Isaac-Reach-Franka）的真实配置——
    文档里过去只写了"是 7 个"，但没有可执行的方式证明。

为什么不开仿真
    AppLauncher 启动 Isaac Sim 要 40 秒并吃掉十几 GB 内存。
    配置是纯 Python dataclass，import 就能拿到全部信息。**先用零成本方式看懂结构，
    再决定要不要花 GPU 时间去训**——这是 C2 与 C3 的分界线。

运行方式（不需要 OMNI_KIT_ACCEPT_EULA，不启动仿真）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    python D:\\work\\embodied-intelligence\\code\\c2_inspect_managers.py
    python D:\\work\\embodied-intelligence\\code\\c2_inspect_managers.py --task Isaac-Reach-Franka-OSC
"""

from __future__ import annotations

import argparse
import sys

MANAGERS = ["observations", "actions", "rewards", "terminations", "events", "commands", "curriculum"]
MANAGER_ROLE = {
    "observations": "把场景状态转成策略输入（obs）",
    "actions": "把策略输出转成关节/夹爪指令",
    "rewards": "把任务目标标量化成 reward",
    "terminations": "判定回合结束（success / timeout）",
    "events": "reset/setup/interval 时的状态重置与域随机化",
    "commands": "生成并管理指令/目标（没有它就没有'要到哪去'）",
    "curriculum": "按训练进度动态调难度/权重",
}


def brief(obj, maxlen: int = 78) -> str:
    s = repr(obj)
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def unwrap(value):
    """把 Isaac Lab 的 _Preset 配置对象解包成它的 default 值。

    为什么要这一步：Isaac Lab 3.0 用 PresetCfg 表示"同一任务的多套配置预设"
    （default / hold / diffik / newton_ik …）。直接 repr 会打印成
    _Preset(default=RewardTermCfg(...)) 一大串，看不出实际的权重和函数名。
    本节要看的是 default 那一套，所以取 default。
    """
    seen = 0
    while type(value).__name__ == "_Preset" and seen < 5:
        value = getattr(value, "default", value)
        seen += 1
    return value


def describe_term(term) -> str:
    """从 TermCfg 里抽出 func / weight / params 的要点。"""
    term = unwrap(term)
    parts = []
    func = getattr(term, "func", None)
    if func is not None:
        parts.append(getattr(func, "__name__", str(func)))
    weight = unwrap(getattr(term, "weight", None))
    if weight is not None:
        parts.append(f"w={weight}")
    params = getattr(term, "params", None)
    if params:
        keys = ",".join(sorted(params.keys()))
        parts.append(f"params[{keys}]")
    return "  ".join(parts)


def inspect(task_id: str) -> int:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  触发任务注册

    try:
        spec = gym.spec(task_id)
    except Exception:
        print(f"[错误] 找不到任务 {task_id}\n可用示例：Isaac-Reach-Franka、Isaac-Reach-Franka-OSC、Isaac-Cartpole")
        return 1

    entry = spec.kwargs.get("env_cfg_entry_point")
    if not entry:
        print(f"[错误] {task_id} 没有 env_cfg_entry_point")
        return 1
    module_path, cls_name = entry.split(":")
    mod = __import__(module_path, fromlist=[cls_name])
    cfg = getattr(mod, cls_name)()

    print("=" * 96)
    print(f"任务: {task_id}")
    print(f"配置类: {entry}")
    print(f"环境类: {spec.entry_point}")
    print("=" * 96)

    # ---- 基本运行参数
    print(f"\n【基本参数】num_envs={cfg.scene.num_envs}  episode_length_s={cfg.episode_length_s}  "
          f"decimation={cfg.decimation}  sim.dt={cfg.sim.dt:.6g}")
    print(f"          控制频率 = 1/(dt*decimation) = {1.0 / (cfg.sim.dt * cfg.decimation):.1f} Hz")

    # ---- 7 个管理器
    present, missing = [], []
    print("\n" + "=" * 96)
    print("【7 个管理器总览】")
    print("=" * 96)
    print(f"{'管理器':<14} {'项数':>5}  职责")
    print("-" * 96)
    for name in MANAGERS:
        mgr = getattr(cfg, name, None)
        if mgr is None:
            missing.append(name)
            print(f"{name:<14} {'—':>5}  ⚠ 本任务未使用")
            continue
        terms = {k: v for k, v in vars(mgr).items()
                 if not k.startswith("_") and not callable(v)}
        present.append(name)
        print(f"{name:<14} {len(terms):>5}  {MANAGER_ROLE[name]}")

    print(f"\n结论：本任务使用 {len(present)}/7 个管理器", end="")
    if missing:
        print(f"，未使用：{', '.join(missing)}")
    else:
        print("（7 个全部使用）")

    # ---- 逐管理器展开
    for name in present:
        mgr = getattr(cfg, name)
        print("\n" + "=" * 96)
        print(f"【{name}】{MANAGER_ROLE[name]}")
        print("=" * 96)
        terms = {k: v for k, v in vars(mgr).items()
                 if not k.startswith("_") and not callable(v)}
        # observations 是分组结构，单独处理
        if name == "observations":
            for group_name, group in terms.items():
                sub = {k: v for k, v in vars(group).items()
                       if not k.startswith("_") and not callable(v)}
                terms_ = {k: unwrap(v) for k, v in sub.items() if hasattr(unwrap(v), "func")}
                print(f"  组 [{group_name}] 共 {len(terms_)} 项 "
                      f"(concatenate_terms={getattr(group, 'concatenate_terms', None)}, "
                      f"enable_corruption={getattr(group, 'enable_corruption', None)})")
                for k, v in terms_.items():
                    print(f"    - {k:<34} {describe_term(v)}")
            continue
        for k, v in terms.items():
            core = unwrap(v)
            if hasattr(core, "func") or hasattr(core, "weight"):
                print(f"  {k:<34} {describe_term(v)}")
            else:
                print(f"  {k:<34} {brief(core)}")

    # ---- 观测维度：手工累加，验证"策略输入多少维"
    print("\n" + "=" * 96)
    print("【策略输入维度如何构成】")
    print("=" * 96)
    print("  说明：维度由各观测项的 term 函数决定，需运行时才知道精确值；")
    print("        这里只列出参与了哪些项，精确维度在 C2 第 2 条验收（新增观测项）时打印。")
    for group_name, group in vars(cfg.observations).items():
        if group_name.startswith("_"):
            continue
        sub = [k for k, v in vars(group).items() if hasattr(v, "func")]
        print(f"  组 [{group_name}]: {', '.join(sub)}")

    print("\n" + "=" * 96)
    print("下一步：把上面的结构抄进总计划 C2 节，并做验收第 2 条（新增一个观测项）。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="C2: 只读解剖 manager-based 任务配置")
    ap.add_argument("--task", default="Isaac-Reach-Franka", help="任务 id")
    args = ap.parse_args()
    return inspect(args.task)


if __name__ == "__main__":
    sys.exit(main())
