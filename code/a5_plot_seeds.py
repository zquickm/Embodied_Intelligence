"""A5 结果出图：3 种子均值±方差曲线 + 超参敏感性表 + 稳定性表

作用
    读 `code/a5_sweep.py` 生成的缓存（默认 `file/assets/a5_runs.json`），产出：
    1. `a5_cartpole_3seed.png`      —— 默认配置 3 个种子的曲线 + 均值线 + ±1 std 阴影
    2. `a5_sensitivity.png`         —— 3 个超参的取值-成绩关系（误差棒 = 种子间 std）
    3. `a5_hyperparam_table.md`     —— 超参敏感性表（三个指标并列）
    4. `a5_stability_table.md`      —— 达标后崩溃的稳定性表

为什么指标要三个，而不是一个
    CartPole 单集上限 500，默认配置的峰值三种子全是 500.0 —— **峰值指标饱和了**，
    任何“够好”的配置都会撞到天花板，用它比超参只会得到“都一样”的假结论。
    所以必须同时看：
      - peak（best avg100）  ：最终能到多高 —— 会饱和，只用来发现“学不会”的配置
      - eps_to_solve         ：多快达到目标 —— 饱和后仍能区分优劣（样本效率）
      - drop（best - final）  ：学到的会不会丢 —— REINFORCE 达标后崩溃的程度
    这是 A5 真正要建立的能力：**指标选错，结论就是错的**。

为什么“崩溃”必须单独看
    实测（2026-10-26，本机）：默认配置 seed1 在第 531 集达到 500 分，
    训练结束只剩 9.3 分（低于随机策略）。只看 final 会得出“默认配置最差(291.0)”
    的错误结论，而它的峰值其实是 500.0。崩溃不是噪声，是算法/步长问题，
    要单独量化（drop 列），不能混进超参比较里。

对整个项目的作用
    - A5 验收：3 种子曲线 + 超参表
    - C2：改奖励后用同一套“峰值 + 样本效率 + 稳定性”三指标对比，避免只看一条曲线
    - 面试：能讲“我发现最终指标会掩盖崩溃，所以改用峰值和样本效率”，比报分数有说服力

运行方式（先跑 a5_sweep.py，再跑这个）
    . D:\\work\\embodied-intelligence\\scripts\\activate.ps1
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\a5_plot_seeds.py
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
# 崩塌判据：峰值比训练结束高这么多分，认定为“达标后崩溃”而不是正常波动
DROP_COLLAPSE = 100.0


def std_of(values) -> float:
    """样本标准差（ddof=1），与 A2 文档的报法一致。

    A2 报的 LunarLander "268.7 ± 9.7" 是 ddof=1（278.4/268.7/259.1 → 9.65），
    ddof=0 会得 7.88。同一份数据两处用不同口径，文档就会自相矛盾。
    样本标准差也是报告里的通常做法：分母 n-1 是对“用 n 个样本估计均值”的自由度修正。
    种子数 < 2 时无法估计离散度，返回 0.0。
    """
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
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
    return sorted({int(r["seed"]) for r in runs
                   if r["task"] == task
                   and abs(r["lr"] - cfg["lr"]) < 1e-12
                   and abs(r["gamma"] - cfg["gamma"]) < 1e-12
                   and int(r["hidden"]) == int(cfg["hidden"])})


def group_stats(runs: list[dict], task: str, cfg: dict) -> dict | None:
    """一个配置的汇总：峰值、样本效率、崩溃幅度、解决率。"""
    rows = [match(runs, task, cfg, s) for s in seeds_of(runs, task, cfg)]
    rows = [r for r in rows if r and math.isfinite(float(r["final"]))]
    if not rows:
        return None
    peaks = [float(r["best"]) for r in rows]
    finals = [float(r["final"]) for r in rows]
    solved_eps = [float(r["solved_at"]) for r in rows if r.get("solved_at")]
    drops = [p - f for p, f in zip(peaks, finals)]
    return dict(
        n=len(rows),
        peak_mean=float(np.mean(peaks)), peak_std=std_of(peaks),
        final_mean=float(np.mean(finals)), final_std=std_of(finals),
        solved_n=len(solved_eps), eps_mean=float(np.mean(solved_eps)) if solved_eps else None,
        eps_std=std_of(solved_eps) if len(solved_eps) > 1 else 0.0,
        drop_mean=float(np.mean(drops)), drop_std=std_of(drops),
        collapsed_n=sum(1 for d in drops if d > DROP_COLLAPSE),
    )


# --------------------------------------------------------------------------- 图 1
def seed_curves(runs: list[dict], task: str, cfg: dict, seeds: list[int]):
    """返回 {seed: (xs, ys)}，只保留所有种子共有、且数值有限的横坐标。"""
    data: dict[int, dict[int, float]] = {}
    for s in seeds:
        r = match(runs, task, cfg, s)
        if r is None or not r.get("curve"):
            continue
        data[s] = {int(p[0]): float(p[1]) for p in r["curve"]}
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
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    n_seed = mat.shape[0]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for s in sorted(curves):
        ax.plot(curves[s][0], curves[s][1], lw=1.0, alpha=0.55, marker="o", ms=3,
                label=f"seed {s}")
    ax.plot(xs, mean, lw=2.2, color="black", label=f"mean of {n_seed} seed(s)")
    if n_seed > 1:
        ax.fill_between(xs, mean - std, mean + std, color="black", alpha=0.15,
                        label="+-1 std across seeds")
    ax.axhline(TASK_LIMIT[task], ls="--", c="gray", lw=1,
               label=f"env max {TASK_LIMIT[task]:.0f}")
    ax.set_xlabel("episode" if task == "cartpole" else "timesteps")
    ax.set_ylabel("avg100 return (100-episode moving average)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")
    return out


# --------------------------------------------------------------------------- 图 2
def plot_sensitivity(runs: list[dict], out: Path) -> Path | None:
    """上下两排：上排 = 峰值（会饱和），下排 = 达标集数（饱和后仍可区分）。"""
    panels = []
    for hyper in ["lr", "gamma", "hidden"]:
        values, peak_m, peak_s, ep_m, ep_s, solved_frac = [], [], [], [], [], []
        for v in SWEEPS[hyper]:
            cfg = dict(DEFAULTS_CP)
            cfg[hyper] = v
            g = group_stats(runs, "cartpole", cfg)
            if g is None:
                continue
            values.append(v)
            peak_m.append(g["peak_mean"]); peak_s.append(g["peak_std"])
            ep_m.append(g["eps_mean"] if g["eps_mean"] is not None else float("nan"))
            ep_s.append(g["eps_std"])
            solved_frac.append(f"{g['solved_n']}/{g['n']}")
        if len(values) >= 2:
            panels.append((hyper, values, peak_m, peak_s, ep_m, ep_s, solved_frac))

    if not panels:
        print("  [skip] 没有 sweep 数据，先跑 python code/a5_sweep.py")
        return None

    ncol = len(panels)
    fig, axes = plt.subplots(2, ncol, figsize=(4.3 * ncol, 7.2), squeeze=False)
    for j, (hyper, values, peak_m, peak_s, ep_m, ep_s, solved_frac) in enumerate(panels):
        xs = np.arange(len(values))

        ax = axes[0][j]
        ax.errorbar(xs, peak_m, yerr=peak_s, fmt="o-", capsize=4, lw=1.6)
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_title(f"{HYPER_LABEL[hyper]} — peak (ceiling!)", fontsize=9)
        ax.set_ylabel("peak avg100")
        ax.grid(alpha=0.3)
        for x, m, f in zip(xs, peak_m, solved_frac):
            ax.annotate(f"{m:.0f}\nsolved {f}", (x, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)
        if all(m >= 0.97 * TASK_LIMIT["cartpole"] for m in peak_m):
            ax.axhspan(0.97 * TASK_LIMIT["cartpole"], TASK_LIMIT["cartpole"] * 1.02,
                       color="orange", alpha=0.12)
            ax.text(0.5, 0.06, "saturated — use lower row", transform=ax.transAxes,
                    ha="center", fontsize=7, color="darkorange")

        ax = axes[1][j]
        ax.errorbar(xs, ep_m, yerr=ep_s, fmt="s-", capsize=4, lw=1.6, color="tab:green")
        ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in values])
        ax.set_title(f"{HYPER_LABEL[hyper]} — episodes to target", fontsize=9)
        ax.set_xlabel("value"); ax.set_ylabel("episodes (lower = faster)")
        ax.grid(alpha=0.3)
        for x, m in zip(xs, ep_m):
            if math.isfinite(m):
                ax.annotate(f"{m:.0f}", (x, m), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)

    fig.suptitle("CartPole REINFORCE: sensitivity — peak saturates at 500, sample efficiency does not",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out.name}")
    return out


# --------------------------------------------------------------------------- 表 1
def verdict_of(g: dict, base: dict) -> tuple[str, str]:
    """给一个取值下判断，返回 (判断, 依据)。

    为什么不能只看 peak：CartPole 峰值饱和到 500，绝大多数配置都撞天花板，
    用 peak 判断只会得到“都一样”。而 eps（达标集数）只在**达标种子上**取均值，
    一个“3 个种子只成功 1 个但成功那次很快”的配置会显得比稳定配置更快
    ——这是幸存者偏差。

    所以按可靠性从高到低排序判断，命中即停：
      1. 解决率（solved_n / n）低于基准  → 有影响（学不会）
      2. 崩溃种子数多于基准              → 有影响（不稳定）
      3. eps 差异超出波动                → 有影响（样本效率）
      4. 都不成立                        → 不显著
    """
    if g["solved_n"] < base["solved_n"]:
        return "有影响", f"解决率 {g['solved_n']}/{g['n']} < 基准 {base['solved_n']}/{base['n']}"
    if g["collapsed_n"] > base["collapsed_n"]:
        return "有影响", f"崩溃 {g['collapsed_n']}/{g['n']} > 基准 {base['collapsed_n']}/{base['n']}"
    if g["eps_mean"] is not None and base["eps_mean"] is not None:
        d = g["eps_mean"] - base["eps_mean"]
        noise = max(g["eps_std"], base["eps_std"])
        if abs(d) > noise:
            return "有影响", f"eps 差 {d:+.0f} > 波动 {noise:.0f}"
    if g["solved_n"] > base["solved_n"] or g["collapsed_n"] < base["collapsed_n"]:
        return "更优", f"解决率/稳定性优于基准"
    return "不显著", "各指标差异在波动范围内"


def write_hyperparam_table(runs: list[dict], out: Path) -> Path | None:
    rows = []
    for hyper in ["lr", "gamma", "hidden"]:
        per_value: dict[float, dict] = {}
        for v in SWEEPS[hyper]:
            cfg = dict(DEFAULTS_CP)
            cfg[hyper] = v
            g = group_stats(runs, "cartpole", cfg)
            if g:
                per_value[v] = g
        base = per_value.get(DEFAULTS_CP[hyper])
        if not base:
            continue
        for v, g in per_value.items():
            is_default = abs(v - DEFAULTS_CP[hyper]) < 1e-12
            if is_default:
                verdict, reason = "默认", "—"
            else:
                verdict, reason = verdict_of(g, base)
            eps = "—" if g["eps_mean"] is None else f"{g['eps_mean']:.0f}"
            rows.append((HYPER_LABEL[hyper], f"{v:g}", f"{g['peak_mean']:.1f}",
                         f"{g['solved_n']}/{g['n']}", eps,
                         f"{g['collapsed_n']}/{g['n']}", f"{g['drop_mean']:.0f}",
                         verdict, reason))

    if not rows:
        print("  [skip] 没有可汇总的数据")
        return None

    lines = [
        "# A5 超参敏感性表（CartPole + REINFORCE，3 种子，固定 1200 集预算）",
        "",
        "> 由 `code/a5_plot_seeds.py` 自动生成，**判断列不要人工改**。",
        ">",
        "> 指标：**peak** = 全程 best avg100（**会饱和到 500，不宜单独用来分优劣**）；",
        "> **solved** = 达标(475)种子数；**eps** = 达标集数（仅统计达标种子，含幸存者偏差）；",
        "> **崩溃** = peak−final > 100 的种子数；**drop** = peak−final 均值。",
        ">",
        "> 判断规则（按可靠性排序，命中即停）：解决率 → 崩溃数 → eps 差异 → 更优/不显著。",
        "",
        "| 超参 | 取值 | peak | solved | eps | 崩溃 | drop | 判断 | 依据 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += [
        "",
        "**怎么读**：peak 一列几乎全是 500，说明这个任务的**上限被环境卡住**，"
        "任何够好的配置都能撞顶——所以真正的差异在 solved / 崩溃 / eps 三列。",
        "eps 越小越快，但它只对达标种子平均：一个 2/3 达标的配置，eps 可能“看起来很快”实则是幸存者偏差，必须配合 solved 一起看。",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {out.name}")
    return out


# --------------------------------------------------------------------------- 表 2
def write_stability_table(runs: list[dict], out: Path) -> Path | None:
    rows = []
    for hyper in ["lr", "gamma", "hidden"]:
        for v in SWEEPS[hyper]:
            cfg = dict(DEFAULTS_CP)
            cfg[hyper] = v
            g = group_stats(runs, "cartpole", cfg)
            if g:
                rows.append((HYPER_LABEL[hyper], f"{v:g}", f"{g['peak_mean']:.1f}",
                             f"{g['final_mean']:.1f}", f"{g['drop_mean']:.1f}",
                             f"{g['collapsed_n']}/{g['n']}"))
    if not rows:
        return None
    lines = [
        "# A5 稳定性表：达标后崩溃（REINFORCE）",
        "",
        "> 由 `code/a5_plot_seeds.py` 自动生成。",
        f"> **drop = peak - final**：峰值减去训练结束时的成绩。drop > {DROP_COLLAPSE:.0f} 记为崩溃。",
        "> 崩溃是算法/步长问题，不是噪声——同一配置内多个种子同时崩溃即视为系统性缺陷。",
        "",
        "| 超参 | 取值 | peak 均值 | final 均值 | drop 均值 | 崩溃种子数 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines += ["", "**为什么单列一张表**：`final` 会把“学到过但丢了”和“根本没学会”混为一谈。"
                  "前者要修稳定性（减步长/减训练集数/早停），后者要改算法或加训练量——处置完全不同。"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  saved {out.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A5: 出图 + 超参敏感性表 + 稳定性表")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--task", choices=["cartpole", "lunarlander", "all"], default="cartpole")
    args = ap.parse_args()

    cache, out_dir = Path(args.cache), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(cache)
    print(f"缓存 {cache} 里共 {len(runs)} 条 run")

    if args.task in ("cartpole", "all"):
        plot_seed_band(runs, "cartpole", DEFAULTS_CP,
                       out_dir / "a5_cartpole_3seed.png",
                       "CartPole-v1 / REINFORCE: default config, 3 seeds")
        plot_sensitivity(runs, out_dir / "a5_sensitivity.png")
        write_hyperparam_table(runs, out_dir / "a5_hyperparam_table.md")
        write_stability_table(runs, out_dir / "a5_stability_table.md")
    if args.task in ("lunarlander", "all"):
        plot_seed_band(runs, "lunarlander", DEFAULTS_LL,
                       out_dir / "a5_lunarlander_3seed.png",
                       "LunarLander-v3 / DQN: baseline, 3 seeds")

    for task, cfg in (("cartpole", DEFAULTS_CP), ("lunarlander", DEFAULTS_LL)):
        if args.task not in (task, "all"):
            continue
        g = group_stats(runs, task, cfg)
        if not g:
            continue
        eps = "—" if g["eps_mean"] is None else f"{g['eps_mean']:.0f}"
        print(f"{task:11s} baseline: peak={g['peak_mean']:.1f}±{g['peak_std']:.1f} "
              f"final={g['final_mean']:.1f}±{g['final_std']:.1f} "
              f"solved={g['solved_n']}/{g['n']} eps={eps} drop={g['drop_mean']:.1f}")


if __name__ == "__main__":
    main()
