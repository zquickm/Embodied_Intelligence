"""A5 结果出图：3 种子均值±方差曲线 + 超参敏感性表

作用
    读 `code/a5_sweep.py` 生成的缓存（默认 `file/assets/a5_runs.json`），产出三样东西：
    1. `a5_cartpole_3seed.png`   —— 同一配置 3 个种子的回报曲线 + 均值线 + ±1 std 阴影
    2. `a5_sensitivity.png`      —— 3 个超参的取值-成绩关系（误差棒 = 种子间 std）
    3. `a5_hyperparam_table.md`  —— 可以直接粘进总计划的超参敏感性表

为什么单独出表，而不只看图
    图只能看趋势。A5 的判断标准是“均值差 > 种子间 std 才算真有影响”，
    这个比较必须用数字做，所以脚本会把结论直接写进表里，避免凭感觉下判断。

对整个项目的作用
    - A5 验收：第 1 条（3 种子曲线）和第 2 条（超参表）的产物就是这三份文件
    - C2：改奖励后把这里的对比逻辑套过去，就是“改奖励前后各 3 个种子”的判据
    - 面试：被问“你怎么知道不是你运气好”，直接给这张表

运行方式（先跑 a5_sweep.py，再跑这个）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\a5_plot_seeds.py
    python D:\\work\\embodied-intelligence\\code\\a5_plot_seeds.py --task lunarlander
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "file" / "assets" / "a5_runs.json"
DEFAULT_OUT = REPO / "file" / "assets"

DEFAULTS_CP = {"lr": 3e-3, "gamma": 0.99, "hidden": 128}
DEFAULTS_LL = {"lr": 5e-4, "gamma": 0.99, "hidden": 256}
SWEEPS = {
    "lr": [1e-3, 3e-3, 1e-2],
    "gamma": [0.90, 0.95, 0.99],
    "hidden": [32, 64, 128],
}
HYPER_LABEL = {"lr": "learning rate", "gamma": "discount gamma", "hidden": "hidden width"}
TASK_LIMIT = {"cartpole": 500.0, "lunarlander": 300.0}


def std_of(values) -> float:
    """样本标准差（ddof=1），与 A2 文档的报法一致。

    为什么必须是 ddof=1 而不是 numpy 默认的 ddof=0：
        A2 文档报的 LunarLander "268.7 ± 9.7" 用的是样本标准差
        （278.4/268.7/259.1 → ddof=1 得 9.65，ddof=0 得 7.88）。
        同一份数据在两处用不同口径，文档就会自相矛盾。
        样本标准差也是论文/报告的通常做法：它估计的是"总体离散度"，
        分母用 n-1 是对 n 个样本估计 1 个均值的自由度修正。
    种子数 < 2 时无法估计离散度，返回 0.0 并在图上退化为一条线。
    """
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1))


def load_runs(cache: Path) -> list[dict]:
    if not cache.exists():
        sys.exit(f"找不到缓存 {cache}\n先跑：python code/a5_sweep.py")
    try:
        return json.loads(cache.read_text(encoding="utf-8")).get("runs", [])
    except Exception as exc:
        sys.exit(f"缓存读不出来：{exc}")


def match(runs: list[dict], task: str, cfg: dict, seed: int) -> dict | None:
    for r in runs:
        if (r["task"] == task and int(r["seed"]) == seed
                and abs(r["lr"] - cfg["lr"]) < 1e-12
                and abs(r["gamma"] - cfg["gamma"]) < 1e-12
                and int(r["hidden"]) == int(cfg["hidden"])):
            return r
    return None


def seeds_of(runs: list[dict], task: str, cfg: dict) -> list[int]:
    out = []
    for s in range(0, 20):
        r = match(runs, task, cfg, s)
        if r is not None and s not in out:
            out.append(s)
    return sorted(out)


def seed_curves(runs: list[dict], task: str, cfg: dict, seeds: list[int]):
    """返回 {seed: (xs, ys)}，只保留所有种子共有、且数值有限的横坐标。"""
    data: dict[int, dict[int, float]] = {}
    for s in seeds:
        r = match(runs, task, cfg, s)
        if r is None or not r.get("curve"):
            continue
        data[s] = {int(p[0]): float(p[1]) for p in r["curve"]}
    if len(data) < len(seeds):
        missing = [s for s in seeds if s not in data]
        print(f"  [warn] 缺种子 {missing} 的曲线，按现有 {len(data)} 个种子画")
    if not data:
        return {}
    common = set.intersection(*(set(d) for d in data.values()))
    common = sorted(x for x in common if all(math.isfinite(d[x]) for d in data.values()))
    if not common:
        return {}
    return {s: (common, [data[s][x] for x in common]) for s in data}


def plot_seed_band(runs: list[dict], task: str, cfg: dict, out: Path, title: str) -> Path | None:
    seeds = seeds_of(runs, task, cfg)
    curves = seed_curves(runs, task, cfg, seeds)
    if not curves:
        print(f"  [skip] {task} 没有基线数据，先跑 sweep --stage baseline")
        return None

    xs = next(iter(curves.values()))[0]
    mat = np.vstack([np.asarray(curves[s][1], dtype=float) for s in sorted(curves)])
    mean = mat.mean(axis=0)
    # 同一集数上的种子间离散度（n<2 时为 0，图上退化为一条均值线）
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    n_seed = mat.shape[0]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for s in sorted(curves):
        ax.plot(curves[s][0], curves[s][1], lw=0.8, alpha=0.35, label=f"seed {s}")
    ax.plot(xs, mean, lw=2.0, color="black", label=f"mean of {n_seed} seed(s)")
    if n_seed > 1:
        ax.fill_between(xs, mean - std, mean + std, color="black", alpha=0.15,
                        label="+-1 std across seeds")
    ax.axhline(TASK_LIMIT[task], ls="--", c="gray", lw=1,
               label=f"env max/threshold {TASK_LIMIT[task]:.0f}")
    ax.set_xlabel("episode" if task == "cartpole" else "timesteps")
    ax.set_ylabel("return" if task == "cartpole" else "eval return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")
    return out


def plot_sensitivity(runs: list[dict], out: Path) -> Path | None:
    panels = []
    for hyper in ["lr", "gamma", "hidden"]:
        values, means, stds, ns = [], [], [], []
        for v in SWEEPS[hyper]:
            cfg = dict(DEFAULTS_CP)
            cfg[hyper] = v
            finals = []
            for s in seeds_of(runs, "cartpole", cfg):
                r = match(runs, "cartpole", cfg, s)
                finals.append(float(r["final"]))
            finals = [f for f in finals if math.isfinite(f)]
            if not finals:
                continue
            values.append(v)
            means.append(float(np.mean(finals)))
            stds.append(std_of(finals))
            ns.append(len(finals))
        if len(values) >= 2:
            panels.append((hyper, values, means, stds, ns))

    if not panels:
        print("  [skip] 没有 sweep 数据，先跑 python code/a5_sweep.py（默认会跑 sweep）")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (hyper, values, means, stds, ns) in zip(axes, panels):
        xs = np.arange(len(values))
        ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4, lw=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_title(HYPER_LABEL[hyper])
        ax.set_xlabel("value")
        ax.set_ylabel("final avg100")
        ax.grid(alpha=0.3)
        for x, m, n in zip(xs, means, ns):
            ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle("CartPole REINFORCE: hyperparameter sensitivity (mean +- std over seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")
    return out


def write_table(runs: list[dict], out: Path) -> Path | None:
    rows = []
    for hyper in ["lr", "gamma", "hidden"]:
        per_value = {}
        for v in SWEEPS[hyper]:
            cfg = dict(DEFAULTS_CP)
            cfg[hyper] = v
            finals = []
            for s in seeds_of(runs, "cartpole", cfg):
                finals.append(float(match(runs, "cartpole", cfg, s)["final"]))
            finals = [f for f in finals if math.isfinite(f)]
            if finals:
                per_value[v] = (float(np.mean(finals)), std_of(finals), len(finals))
        if not per_value:
            continue
        default_v = DEFAULTS_CP[hyper]
        if default_v not in per_value:
            continue
        base_mean, base_std, _ = per_value[default_v]
        for v, (mean, std, n) in per_value.items():
            diff = mean - base_mean
            threshold = max(std, base_std)
            verdict = "有影响" if abs(diff) > threshold else "不显著"
            if abs(v - default_v) < 1e-12:
                verdict = "默认"
                diff = 0.0
            rows.append((HYPER_LABEL[hyper], f"{v:g}", f"{mean:.1f}", f"±{std:.1f}",
                         n, f"{diff:+.1f}", verdict))

    if not rows:
        print("  [skip] 没有可汇总的数据")
        return None

    lines = [
        "# A5 超参敏感性表（CartPole + REINFORCE）",
        "",
        "> 由 `code/a5_plot_seeds.py` 自动生成。判据：|均值差| > max(该取值 std, 默认 std) 记为“有影响”。",
        "> 均值与 std 均为跨种子统计；std 用样本标准差（ddof=1），与 A2 文档口径一致。",
        "> CartPole 单集上限 500；指标为固定预算的 final avg100（近 100 集平均）。",
        "",
        "| 超参 | 取值 | 均值 | 种子间 std | 种子数 | 相对默认 | 判断 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", "**读法**：判断列是脚本按上面的规则算的，不是人工填的；"
                  "“不显著”不等于没影响，只说明在当前种子数和预算下测不出来。"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {out.name}")
    print("\n" + "\n".join(lines[3:]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A5: 出图 + 超参敏感性表")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--task", choices=["cartpole", "lunarlander", "all"], default="cartpole")
    args = ap.parse_args()

    cache = Path(args.cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(cache)
    print(f"缓存 {cache} 里共 {len(runs)} 条 run")

    if args.task in ("cartpole", "all"):
        plot_seed_band(runs, "cartpole", DEFAULTS_CP,
                       out_dir / "a5_cartpole_3seed.png",
                       "CartPole-v1 / REINFORCE: 3 seeds (mean +- 1 std)")
        plot_sensitivity(runs, out_dir / "a5_sensitivity.png")
        write_table(runs, out_dir / "a5_hyperparam_table.md")
    if args.task in ("lunarlander", "all"):
        plot_seed_band(runs, "lunarlander", DEFAULTS_LL,
                       out_dir / "a5_lunarlander_3seed.png",
                       "LunarLander-v3 / DQN: 3 seeds (eval mean +- 1 std)")

    # 基线种子成绩，直接抄进实验记录
    for task, cfg in (("cartpole", DEFAULTS_CP), ("lunarlander", DEFAULTS_LL)):
        if args.task not in (task, "all"):
            continue
        finals = [(s, float(match(runs, task, cfg, s)["final"]))
                  for s in seeds_of(runs, task, cfg)]
        finals = [(s, v) for s, v in finals if math.isfinite(v)]
        if finals:
            vals = np.array([v for _, v in finals])
            pairs = " / ".join(f"{s}:{v:.1f}" for s, v in finals)
            print(f"{task:11s} seeds {pairs}  mean={vals.mean():.1f} std={std_of(vals):.1f} (ddof=1)")


if __name__ == "__main__":
    main()
