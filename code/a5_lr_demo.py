"""A5 讲解素材：学习率 lr 到底在做什么（一维演示）

作用
    用一个初中就会的抛物线 f(w) = w²，演示梯度下降里 lr 的作用。
    不用神经网络、不用 CartPole，只有 1 个参数 w，所以能一眼看出
    "每步走多远"到底是什么，以及为什么 lr 太大会发散。

为什么单独写一个 demo
    CartPole 里 lr 影响的是一百万个参数，肉眼看不见。
    把问题降到 1 个参数，规律完全一样，但可以打印出每一步的数字。

数学（三行就够）
    f(w) = w²            谷底在 w=0，最小值 f=0
    f'(w) = 2w           梯度：告诉我"往哪边下降"
    w ← w - lr * 2w      梯度下降：往梯度反方向走，走多远由 lr 决定

    把更新写成 w_new = w * (1 - 2*lr)，所以 lr 决定这个乘数：
        lr < 0.5  →  乘数在 (0,1)      →  稳定收敛，但 lr 越小越慢
        lr = 0.5  →  乘数 = 0          →  一步到位（这是本例的最优）
        lr > 0.5  →  乘数在 (-1,0)     →  来回震荡，但幅度递减（还收敛）
        lr = 1.0  →  乘数 = -1         →  左右横跳，永不收敛
        lr > 1.0  →  |乘数| > 1        →  越跳越远，发散

运行方式
    python D:\\work\\embodied-intelligence\\code\\a5_lr_demo.py
"""

import numpy as np

FEATURES = {
    # lr: 说明
    0.01: "太小：方向对，但 20 步只走了一点点",
    0.1: "合适：20 步已经贴近谷底",
    0.5: "本例最优：1 步直接到谷底（这是运气，因为 f=w² 的特殊形状）",
    0.9: "偏大：左右震荡着靠近，能看到正负交替",
    1.0: "临界：永远在 -1 和 1 之间横跳，永远不收敛",
    1.1: "发散：每跳一次离谷底更远（CartPole 里 lr=1e-2 就是这种）",
}


def run(lr: float, steps: int = 20, w0: float = 1.0):
    """从 w0 出发，做 steps 次梯度下降，返回每步的 w 和 f(w)。"""
    w = w0
    ws = [w]
    for _ in range(steps):
        grad = 2 * w          # f'(w) = 2w
        w = w - lr * grad     # 这就是 "每步走多远" 的全部内容
        ws.append(w)
    return np.array(ws)


def main() -> None:
    print("目标：找 f(w)=w² 的谷底（w=0）。起点 w=1.0，看 lr 怎样决定过程。\n")
    print(f"{'lr':>6} | {'第1步 w':>10} {'第5步 w':>10} {'第20步 w':>10} | "
          f"{'第20步 f(w)':>11} | 现象")
    print("-" * 96)

    for lr, note in FEATURES.items():
        ws = run(lr)
        f20 = ws[20] ** 2
        print(f"{lr:>6g} | {ws[1]:>10.4f} {ws[5]:>10.4f} {ws[20]:>10.4f} | "
              f"{f20:>11.4f} | {note}")

    print("\n" + "=" * 96)
    print("把 lr=0.9 的完整轨迹打出来，看'震荡'长什么样（符号正负交替）：\n")
    ws = run(0.9, steps=10)
    for i, w in enumerate(ws):
        bar = "█" * int(abs(w) * 30)
        sign = "→" if w >= 0 else "←"
        print(f"  第{i:>2}步  w={w:>8.4f}  f={w*w:>8.4f}  {sign} {bar}")

    print("\n" + "=" * 96)
    print("对照：lr=0.1（小步）走 20 步，还差多远？")
    ws_small = run(0.1)
    ws_big = run(0.5)
    print(f"  lr=0.1 走 20 步: w={ws_small[20]:.6f}  f={ws_small[20]**2:.8f}  （很接近但没到）")
    print(f"  lr=0.5 走  1 步: w={ws_big[1]:.6f}  f={ws_big[1]**2:.8f}  （一步到）")

    print("\n" + "=" * 96)
    print("结论：")
    print("  1. lr 就是'沿着梯度方向迈多大一步'。梯度只给方向，lr 给距离。")
    print("  2. lr 小 = 慢但稳；lr 大 = 快但会震荡；lr 过大 = 越过谷底越弹越远（发散）。")
    print("  3. 最优 lr 不是固定的：它取决于问题形状。本例 0.5 最好，")
    print("     CartPole 上实测 3e-3 最好（见 a5_hyperparam_table.md），差 100 倍以上很正常。")
    print("  4. CartPole 里 lr=1e-2 让 3/3 种子全部崩溃，原因和上面 lr=1.1 一样：每步跨太大。")


if __name__ == "__main__":
    main()
