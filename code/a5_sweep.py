"""A5 实验方法论：多种子 + 超参敏感性的批量运行器

作用
    批量调用 A2 的两个训练脚本（`code/a2_reinforce.py`、`code/a2_dqn.py`），
    把每个“配置 × 种子”的训练曲线点和最终成绩收进一份缓存 JSON，
    供 `a5_plot_seeds.py` 画均值±方差图和超参敏感性表。

为什么单独写这个文件，而不是去改 a2 脚本
    a2 的两个脚本保持“教具”的单一职责：只训练、只打印。
    批量运行、缓存、去重、汇总属于“实验工程”，正是 A5 要建立的习惯；
    放在这里既能复用 a2 的算法实现，又不污染学习代码。

对整个项目的作用
    - A5：3 种子曲线、超参表、实验记录的数据来源（唯一数据源）
    - C2：改奖励后要用同一套办法对比 3 个种子，把 `run_one` 里的命令换成
      Isaac Lab 的 train 即可复用
    - 面试：能讲清“我怎么控制变量、怎么判断差异是不是噪声”

运行方式（在 D:\\isaac\\work 下跑，不要在仓库目录跑）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    Set-Location D:\\isaac\\work

    # 1) 先看要跑什么，不实际训练
    python D:\\work\\embodied-intelligence\\code\\a5_sweep.py --dry-run

    # 2) 默认：CartPole 基线 3 种子 + 3 个超参各 2 个非默认值，共 21 个配置
    python D:\\work\\embodied-intelligence\\code\\a5_sweep.py

    # 3) 只跑基线（只想先拿 3 种子曲线时用）
    python D:\\work\\embodied-intelligence\\code\\a5_sweep.py --stage baseline

    # 4) LunarLander 基线（慢，建议放后台）
    python D:\\work\\embodied-intelligence\\code\\a5_sweep.py --task lunarlander --stage baseline

设计说明
    - 缓存键 = (task, lr, gamma, hidden, seed, budget)。同一个配置重复跑会被跳过，
      默认值（lr=3e-3 / gamma=0.99 / hidden=128）只在 baseline 跑一次，
      sweep 里不再重复——这样 21 个配置覆盖了 27 个“表项”。
    - 每跑完一个就落盘一次缓存：中途 Ctrl+C 或断电，重启后自动续跑。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIR = REPO / "code"
DEFAULT_CACHE = REPO / "file" / "assets" / "a5_runs.json"
DEFAULT_WORKDIR = Path(r"D:\isaac\work\a5")

# 基线配置：与 A2 脚本里的默认值一致，改这里等于改 A2 的结论，慎重
DEFAULTS_CP = {"lr": 3e-3, "gamma": 0.99, "hidden": 128}
DEFAULTS_LL = {"lr": 5e-4, "gamma": 0.99, "hidden": 256}

# 超参对照的取值：默认值在中间或一侧，其余由 baseline 覆盖
SWEEPS = {
    "lr": [1e-3, 3e-3, 1e-2],
    "gamma": [0.90, 0.95, 0.99],
    "hidden": [32, 64, 128],
}

# a2_reinforce.py 每 100 集打印：  ep  100 avg100= 123.4 last= 50.0   12s
RE_CP_CURVE = re.compile(r"^ep\s+(\d+)\s+avg100=\s*(-?[\d.]+)", re.MULTILINE)
# a2_dqn.py 每次评估打印：        [eval] step   10000 mean  -120.3 +-  30.1
RE_LL_CURVE = re.compile(r"\[eval\]\s+step\s+(\d+)\s+mean\s+(-?[\d.]+)\s+\+-\s+([\d.]+)")


# ----------------------------------------------------------------------------- 缓存
def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = data.get("runs", [])
        return runs if isinstance(runs, list) else []
    except Exception as exc:  # 缓存坏了不该阻塞实验
        print(f"[warn] 缓存读不出来（{exc}），当作空缓存处理: {path}")
        return []


def save_cache(path: Path, runs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # 原子替换：写一半断电也不会毁掉旧缓存


def run_key(task, lr, gamma, hidden, seed, budget) -> tuple:
    return (task, round(float(lr), 8), round(float(gamma), 8), int(hidden), int(seed), int(budget))


def find_run(runs: list[dict], cfg: dict) -> dict | None:
    k = run_key(cfg["task"], cfg["lr"], cfg["gamma"], cfg["hidden"], cfg["seed"], cfg["budget"])
    for r in runs:
        if run_key(r["task"], r["lr"], r["gamma"], r["hidden"], r["seed"], r["budget"]) == k:
            return r
    return None


# ----------------------------------------------------------------------------- 计划
def build_plan(args) -> list[dict]:
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    plan: list[dict] = []

    if args.task in ("cartpole", "all"):
        if args.stage in ("baseline", "all"):
            for s in seeds:
                plan.append(dict(task="cartpole", seed=s, budget=args.episodes, **DEFAULTS_CP))
        if args.stage in ("sweep", "all"):
            for hyper in args.hyper:
                for v in SWEEPS[hyper]:
                    if abs(v - DEFAULTS_CP[hyper]) < 1e-12:
                        continue  # 默认值已由 baseline 覆盖，不重复训练
                    cfg = dict(DEFAULTS_CP)
                    cfg[hyper] = v
                    for s in seeds:
                        plan.append(dict(task="cartpole", seed=s, budget=args.episodes, **cfg))

    if args.task in ("lunarlander", "all") and args.stage in ("baseline", "all"):
        for s in seeds:
            plan.append(dict(task="lunarlander", seed=s, budget=args.timesteps, **DEFAULTS_LL))

    seen, uniq = set(), []
    for c in plan:
        k = run_key(c["task"], c["lr"], c["gamma"], c["hidden"], c["seed"], c["budget"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def describe(cfg: dict) -> str:
    return (f"{cfg['task']:11s} seed={cfg['seed']} lr={cfg['lr']:g} "
            f"gamma={cfg['gamma']:g} hidden={cfg['hidden']} budget={cfg['budget']}")


# ----------------------------------------------------------------------------- 执行
def run_one(cfg: dict, workdir: Path) -> dict:
    if cfg["task"] == "cartpole":
        cmd = [
            sys.executable, str(CODE_DIR / "a2_reinforce.py"),
            "--seed", str(cfg["seed"]),
            "--episodes", str(cfg["budget"]),
            "--lr", str(cfg["lr"]),
            "--gamma", str(cfg["gamma"]),
            "--hidden", str(cfg["hidden"]),
            "--out_dir", str(workdir),
        ]
    else:
        cmd = [
            sys.executable, str(CODE_DIR / "a2_dqn.py"),
            "--seed", str(cfg["seed"]),
            "--timesteps", str(cfg["budget"]),
            "--out_dir", str(workdir),
        ]

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    wall = time.time() - t0
    out = proc.stdout or ""
    if proc.returncode != 0:
        tail = (out[-1200:] + "\n" + (proc.stderr or "")[-1200:]).strip()
        raise RuntimeError(f"训练失败 exit={proc.returncode}\n{tail}")

    final = None
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                final = json.loads(s)
                break
            except json.JSONDecodeError:
                continue
    if final is None:
        raise RuntimeError("没解析到最终结果行（脚本输出格式变了？）:\n" + out[-1200:])

    if cfg["task"] == "cartpole":
        curve = [[int(m.group(1)), float(m.group(2)), None] for m in RE_CP_CURVE.finditer(out)]
        res = dict(final=float(final["final_avg100"]), best=float(final["best_avg100"]),
                   std=None, solved_at=final.get("solved_at"), solved=bool(final["solved"]))
    else:
        curve = [[int(m.group(1)), float(m.group(2)), float(m.group(3))]
                 for m in RE_LL_CURVE.finditer(out)]
        res = dict(final=float(final["final_eval_mean"]), best=float(final["best_eval_mean"]),
                   std=float(final.get("final_eval_std") or 0.0),
                   solved_at=None, solved=bool(final["solved"]))

    if not curve:
        raise RuntimeError("解析到 0 个曲线点，检查 a2 脚本的输出格式")

    # 训练发散时 final 可能是 NaN/Inf（例如 lr 太大直接把策略打崩）。
    # 这种 run 不能进缓存：否则均值、std、图全被污染，事后分不清"跑坏了"和"没跑"。
    for k in ("final", "best"):
        if not math.isfinite(res[k]):
            raise RuntimeError(f"{k} 不是有限值（训练发散/NaN），该配置视为失败")

    res.update(task=cfg["task"], lr=cfg["lr"], gamma=cfg["gamma"], hidden=cfg["hidden"],
               seed=cfg["seed"], budget=cfg["budget"],
               wall_s=round(wall, 1), curve=curve)
    return res


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="A5: 多种子 + 超参敏感性批量实验")
    ap.add_argument("--task", choices=["cartpole", "lunarlander", "all"], default="cartpole")
    ap.add_argument("--stage", choices=["baseline", "sweep", "all"], default="all")
    ap.add_argument("--hyper", nargs="+", choices=list(SWEEPS), default=list(SWEEPS),
                    help="只跑指定超参的 sweep")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=1200, help="CartPole 训练集数（固定预算才能横向比）")
    ap.add_argument("--timesteps", type=int, default=400_000, help="LunarLander 训练步数")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                    help="中间产物目录（ckpt/png），不要放仓库里")
    ap.add_argument("--force", action="store_true", help="忽略缓存重跑")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = build_plan(args)
    cache_path = Path(args.cache)
    runs = load_cache(cache_path)
    todo = [c for c in plan if args.force or find_run(runs, c) is None]

    print(f"计划 {len(plan)} 个配置；缓存里已有 {len(plan) - len(todo)} 个；待跑 {len(todo)} 个")
    print(f"缓存文件: {cache_path}")
    if args.dry_run:
        for c in todo:
            print("  " + describe(c))
        print("\n（dry-run：没有训练。去掉 --dry-run 开始跑。）")
        return 0

    if not todo:
        print("全部配置已完成，直接去画图：python code/a5_plot_seeds.py")
        return 0

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    ok = fail = 0

    for i, cfg in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {describe(cfg)}", flush=True)
        try:
            res = run_one(cfg, workdir)
        except Exception as exc:
            fail += 1
            print(f"  [失败] {exc}\n", flush=True)
            continue
        runs.append(res)
        save_cache(cache_path, runs)
        ok += 1
        eta = (time.time() - t_all) / ok * (len(todo) - i)
        print(f"  -> final={res['final']:.1f} best={res['best']:.1f} "
              f"wall={res['wall_s']:.0f}s  预计剩余 {eta / 60:.0f} 分钟", flush=True)

    print(f"\n完成 {ok} 个，失败 {fail} 个，总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"缓存已写入 {cache_path}")
    print("下一步：python D:\\work\\embodied-intelligence\\code\\a5_plot_seeds.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
