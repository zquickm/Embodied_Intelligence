"""A1：手写 MNIST 训练循环（不依赖任何现成 Trainer）

作用
    一个完整、可直接运行的监督学习训练循环，包含：Dataset/DataLoader、
    训练/验证/测试三划分、AdamW 优化器、混合精度（AMP）、
    checkpoint 保存与按最优验证集加载、loss 曲线绘制。

为什么会有这个文件
    A1 的验收标准要求"不依赖现成 Trainer，手写 MNIST 训练到 >98%"。
    目的是把"训练一个网络"拆成可以逐行解释的动作，而不是调一个 `.fit()`。
    后面 A3 手写 PPO、D1 训 VAE、D2/D3 训 RSSM，用的都是这同一套骨架，
    只是把 loss 从单目标换成多目标。

对整个项目的作用
    - A3/A4：PPO 与 SAC 的训练循环直接复用本文件的骨架
    - D2/D3：世界模型是多目标联合优化，先在这里掌握"单 loss 循环"再扩展
    - A5：checkpoint 与多种子参数（--seed）是后续多种子实验的基础设施
    - C2/C4：理解"一次 Isaac Lab 训练迭代里发生了什么"需要这份底层认知

运行方式（不要在仓库目录里跑，日志/产物会污染 git）
    Set-Location D:\\isaac\\work
    python D:\\work\\embodied-intelligence\\code\\a1_train_loop.py --seed 0 --epochs 6
    python D:\\work\\embodied-intelligence\\code\\a1_train_loop.py --seed 0 --epochs 6 --amp

已实测结果（2026-10-21，6 epoch，3 个种子）
    seed 0: test acc 0.9907
    seed 1: test acc 0.9908
    seed 2: test acc 0.9897
    均值 +- 标准差: 0.9904 +- 0.0006
    过拟合点: 第 7 epoch 附近（15 epoch 实验，val loss 从 0.0305 回升到 0.0508）
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口后端：不弹出绘图窗口，直接存文件
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """两层卷积 + 两层全连接的 MNIST 分类器（约 42 万参数）。

    为什么用这个结构而不是更深的网络：
        A1 的重点是训练循环，不是刷精度。这个规模 6 个 epoch 就能到 99%，
        单 epoch 只要 3.5 秒，可以反复改循环做对比实验。
        换成 ResNet 会把时间花在等训练上，偏离本阶段目标。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def build_loaders(args):
    """构造 train / val / test 三个 DataLoader。

    为什么要从训练集里切出验证集：
        MNIST 官方的 test 集只能用来"最后报一次成绩"。如果拿它来挑最优
        epoch，等于用测试集调超参，报出来的精度是虚高的。
        所以从 60000 张训练集里切 5000 张做验证，test 集全程不参与选择。

    为什么 num_workers=0：
        Windows 用 spawn 启动子进程，num_workers>0 时若脚本缺少
        if __name__ == "__main__" 保护会无限递归创建进程。
        MNIST 数据小（每 epoch 读盘不是瓶颈），单进程反而更稳。
    """
    tf = transforms.Compose([
        transforms.ToTensor(),                        # PIL -> [0,1] float tensor, (1,28,28)
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 全局均值/标准差，让输入零中心
    ])

    full_train = datasets.MNIST(args.data_root, train=True, download=True, transform=tf)

    # 用带种子的 generator，保证同一种子下 train/val 划分固定（可复现的前提）
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        full_train, [len(full_train) - args.val_size, args.val_size], generator=g
    )

    test_ds = datasets.MNIST(args.data_root, train=False, download=True, transform=tf)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在给定数据集上算平均 loss 与准确率。

    为什么加 @torch.no_grad()：
        评估不需要梯度。不关掉的话，PyTorch 会为每个 batch 建计算图，
        显存白白增长，速度也慢。这是最容易被忽略的一行。

    为什么要 model.eval()：
        切换 Dropout（关闭）和 BatchNorm（用滑动统计量而非当前 batch）
        的行为。本模型没用这两层，但评估函数必须写成通用形式——
        以后给模型加 BN 时，忘了 eval() 会得到错误的评估结果。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        # 乘 batch 大小再累加：最后除以样本总数才是真正的"按样本加权"平均
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    model.train()  # 恢复训练模式，避免调用方忘了切回来
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# 单轮训练
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """跑完一个 epoch，返回训练集平均 loss。

    这五行是本阶段最核心的东西，顺序不能错：
        1. optimizer.zero_grad()  清掉上一步的梯度
        2. 前向 model(x)          得到预测
        3. criterion(...)         算损失
        4. scaler.scale(loss).backward()  反向传播求梯度
        5. scaler.step / scaler.update    更新参数
    """
    model.train()
    running_loss, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # (1) 清梯度。PyTorch 的 .grad 是【累加】语义，不是覆盖：
        #     backward() 把新梯度【加到】已有 .grad 上。不清零的话，
        #     第 N 步的梯度 = 前 N 步梯度之和，更新幅度越来越大且方向错乱。
        #     set_to_none=True 直接把 .grad 置为 None 而不是填 0，
        #     省一次显存写入，也让"有没有梯度"可判断。
        optimizer.zero_grad(set_to_none=True)

        # (2)(3) 前向 + 损失。autocast 让矩阵乘/卷积走 fp16，
        #        其余算子（loss、softmax 等）仍用 fp32 保数值稳定。
        #        use_amp=False 时 autocast 是空操作，所以不用写 if/else 两套。
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(x), y)

        # (4) 反向。GradScaler 先把 loss 乘一个大数（初始 65536）再 backward，
        #     避免 fp16 下溢：fp16 能表示的最小正规数约 6e-5，
        #     小梯度会变成 0，训练无声停滞。use_amp=False 时这些调用是空操作。
        scaler.scale(loss).backward()

        # (5) 更新。step() 内部先把梯度除回原尺度、检查有没有 inf/nan，
        #     正常才 optimizer.step()；update() 根据是否溢出调整 scale 因子。
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        seen += x.size(0)
    return running_loss / seen


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="A1: hand-written MNIST training loop")
    p.add_argument("--seed", type=int, default=0, help="随机种子；A5 要求至少 3 个")
    p.add_argument("--epochs", type=int, default=6, help="6 个 epoch 已足够超过 98%")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_size", type=int, default=5000, help="从 60000 训练集里切出的验证集大小")
    p.add_argument("--data_root", default=r"D:\isaac\work\data", help="数据集目录，不进 git")
    p.add_argument("--out_dir", default=r"D:\isaac\work", help="产物目录，不进 git")
    p.add_argument("--amp", action="store_true", help="开启混合精度（小模型上可能反而变慢）")
    args = p.parse_args()

    # 固定所有随机源，保证同种子可复现
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seed={args.seed} amp={args.amp}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, test_ld = build_loaders(args)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # AdamW 而非 SGD：对学习率不敏感，A1 阶段不需要调 lr 就能到 99%。
    # 为什么用 AdamW 而不是 Adam：AdamW 把权重衰减从梯度里解耦出来，
    # 是现在的默认选择（Adam 的 L2 正则和自适应学习率会互相干扰）。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_ep, ckpt = 0.0, 0, out / f"a1_mnist_seed{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_ld, criterion, optimizer, scaler, device, args.amp)
        va_loss, va_acc = evaluate(model, val_ld, criterion, device)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)

        mark = ""
        if va_acc > best_acc:  # 按验证集精度保存最优，而不是按最后一个 epoch
            best_acc, best_ep = va_acc, epoch
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_acc": va_acc},
                ckpt,
            )
            mark = "  <- best"

        print(f"epoch {epoch}/{args.epochs} | {time.time()-t0:5.1f}s | "
              f"train {tr_loss:.4f} | val {va_loss:.4f} | val_acc {va_acc:.4f}{mark}")

    # 加载最优 checkpoint 再测 test：这是唯一一次用到测试集
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te_loss, te_acc = evaluate(model, test_ld, criterion, device)
    print(f"best epoch {ck['epoch']} (val_acc {ck['val_acc']:.4f}) -> TEST acc {te_acc:.4f}")

    # 画 train/val loss 曲线，并标出最优 epoch 的位置
    es = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(es, hist["train_loss"], "o-", label="train loss")
    plt.plot(es, hist["val_loss"], "s-", label="val loss")
    plt.axvline(best_ep, ls="--", c="gray", label=f"best val_acc @ep{best_ep}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = out / f"a1_loss_seed{args.seed}.png"
    plt.savefig(fig, dpi=130)
    plt.close()

    print(f"saved: {ckpt.name}  {fig.name}")
    # 打一行 JSON，方便 A5 用脚本把多个种子的结果汇总成表
    print(json.dumps({
        "seed": args.seed, "amp": args.amp, "epochs": args.epochs,
        "test_acc": round(te_acc, 4), "best_val_acc": round(best_acc, 4), "best_epoch": best_ep,
        "train_loss": [round(v, 4) for v in hist["train_loss"]],
        "val_loss": [round(v, 4) for v in hist["val_loss"]],
    }))


if __name__ == "__main__":
    main()
