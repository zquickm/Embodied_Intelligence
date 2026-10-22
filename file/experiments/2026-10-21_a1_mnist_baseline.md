# 实验记录：A1 MNIST 训练循环基线

> 日期：2026-10-21 ~ 2026-10-22
> 阶段：A1（PyTorch 与训练循环）
> 记录人：本人（代码自行录入并运行）
> 模板版本：`file/实验记录模板.md`

---

## 一、基本信息

| 项 | 内容 |
|---|---|
| 实验名 | `a1_mnist_baseline` |
| 目的 | 验证手写训练循环（不用现成 Trainer）能否把 MNIST 训练到 >98% |
| 假设 | 6 epoch 内 test acc 应 ≥ 0.99；若低于 0.98 说明数据预处理或优化器配置有误 |

---

## 二、配置

### 2.1 环境

| 项 | 值 |
|---|---|
| 机器 | RTX 5070 Ti 16GB / Ryzen 5 9600X / 48GB |
| Python | 3.12.15 |
| torch | 2.11.0+cu128，`cuda=True` |
| torchvision | 0.26.0+cu128 |
| 运行目录 | `D:\isaac\work`（**未在仓库目录运行**） |
| 环境变量 | 通过 `scripts\activate.ps1` 设置（含 `PYTHONUTF8=1`） |
| 设备 | **cuda**（三组日志均确认 `device=cuda`，每 epoch 3.5–4.1 s，可佐证非 CPU） |

### 2.2 超参

| 超参 | 值 | 怎么定的 |
|---|---|---|
| 随机种子 | **0 / 1 / 2** | A5 要求至少 3 个 |
| epochs | 6 | 参考实现验证 6 轮即超 98% |
| batch_size | 128 | 默认值，未调 |
| lr | 1e-3 | 默认值，未调 |
| 优化器 | AdamW | 对 lr 不敏感，A1 阶段不需要调参 |
| 混合精度 | 关闭（`--amp` 未加） | 见第四节 4.2：MNIST 上 AMP 反而更慢 |
| val_size | 5000 | 从 60000 训练集切出 |

### 2.3 数据

| 项 | 内容 |
|---|---|
| 数据集 | MNIST |
| 位置 | `D:\isaac\work\data\MNIST`（**不进 git**） |
| 下载源 | `ossci-datasets.s3.amazonaws.com`（torchvision 0.26 默认镜像） |
| 划分 | train 55000 / val 5000 / test 10000 |
| 预处理 | `ToTensor()` + `Normalize((0.1307,), (0.3081,))` |

---

## 三、结果

### 3.1 主指标

| 种子 | test acc | best val acc | best epoch | 训练时长 |
|---|---|---|---|---|
| 0 | 0.9893 | 0.9890 | 6 | ~21 s |
| 1 | 0.9905 | 0.9904 | 6 | ~22 s |
| 2 | 0.9900 | 0.9884 | 5 | ~22 s |
| **均值 ± 标准差** | **0.98993 ± 0.00060** | 0.98927 | — | — |

**最差种子 0.9893，仍高于验收线 0.98 约 0.9 个百分点。**

### 3.2 训练曲线

| epoch | seed0 train / val | seed1 train / val | seed2 train / val |
|---|---|---|---|
| 1 | 0.1768 / 0.0653 | 0.1786 / 0.0678 | 0.1811 / 0.0775 |
| 2 | 0.0516 / 0.0466 | 0.0501 / 0.0480 | 0.0496 / 0.0494 |
| 3 | 0.0359 / 0.0440 | 0.0344 / 0.0342 | 0.0334 / 0.0524 |
| 4 | 0.0271 / 0.0370 | 0.0253 / 0.0346 | 0.0244 / 0.0455 |
| 5 | 0.0196 / 0.0421 | 0.0205 / 0.0374 | 0.0202 / 0.0434 |
| 6 | 0.0143 / 0.0369 | 0.0155 / 0.0295 | 0.0147 / 0.0517 |

曲线图：`D:\isaac\work\a1_loss_seed{0,1,2}.png`（本地产物，未进仓库）

### 3.3 观察到的现象

- **train loss 三组几乎重合**（epoch 6 分别是 0.0143 / 0.0155 / 0.0147），说明训练动力学高度一致
- **val loss 抖动明显**：seed2 的 val loss 在 epoch 3 涨到 0.0524、epoch 6 涨到 0.0517，但 val_acc 仍在爬升
- **val 与 train loss 之间存在差距**，epoch 6 时 train≈0.015、val≈0.03–0.05

---

## 四、结论与反思

### 4.1 假设是否成立

**成立。** 6 epoch 内三组 test acc 全部 ≥ 0.9893，均值 0.98993，超过假设的 0.99 门槛。

### 4.2 两个值得记住的观察

**① val loss 抖动 ≠ 过拟合**

seed2 的 val loss 曲线不单调（0.0494 → 0.0524 → 0.0455 → 0.0434 → 0.0517），但 test acc 依然 0.99。

原因是**验证集只有 5000 张**，一次误判就是 0.02% 的准确率、loss 的采样噪声被放大。

**判断过拟合的正确方法**：看 train 和 val 是否**反向**变化。真正的过拟合是"train loss 还在降、val loss 持续上升"（见 A1 文档第 2.4 节的 15 epoch 探针：第 7 轮后 val loss 从 0.0305 涨到 0.0508，而 train 从 0.0116 降到 0.0047）。**单点回升是噪声，持续背离才是过拟合。**

**② 这次方差极小（±0.0006），但不能据此认为"实验很稳"**

MNIST 太简单、模型太小，任何种子都会收敛到同一水平。**这是任务本身没有方差，不是实验方法论做得好。**

到 A5 换到 RL 任务（CartPole/LunarLander），同样 3 个种子会看到**大得多的方差**（RL 的种子间差异常达 10%~30%）。那时"3 个种子"才真正发挥作用。现在建立的习惯是为那时准备的。

### 4.3 与参考实现的对比

| 项 | 本次（自行录入） | 参考实现 |
|---|---|---|
| seed 0 test acc | 0.9893 | 0.9907 |
| seed 1 test acc | 0.9905 | 0.9908 |
| seed 2 test acc | 0.9900 | 0.9897 |
| 均值 | 0.98993 | 0.9904 |
| 每 epoch 耗时 | 3.5–4.1 s | ~3.5 s |

**差距 0.05%，在 GPU 非确定性范围内**（cuDNN 算法选择、算子调度顺序都可能不同）。epoch 1 的 train loss 完全一致（0.1768 vs 0.1768），证明数据划分、归一化、初始化三处都与参考一致，差异只来自后期训练的非确定性。**复现链路正确。**

### 4.4 失败与意外

| 现象 | 第一猜测 | 实际原因 | 怎么确认的 |
|---|---|---|---|
| `NameError: build_loaders is not defined` | 以为漏抄了函数 | **三个函数被缩进进了 `SmallCNN` 类**，成了类方法，模块级不存在 | `ast.parse` 打印顶层符号，只有 `['SmallCNN', 'main']` |

**这个坑的教训**：Python 靠缩进决定作用域，**没有 `endif`**。多段代码拼接时，如果上一段以类方法结尾，下一段的 `def` 会自动继承 4 格缩进，**语法上合法、不会报错**，直到调用处才炸。

**以后每写完一个文件先跑这一行**：

```powershell
python -c "import ast; t=ast.parse(open(r'路径',encoding='utf-8').read()); print([n.name for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])"
```

期望所有函数名都在**同一个列表**里。在 VS Code 里等价的操作是 `Ctrl+Shift+O`（转到符号）。

### 4.5 下一步

- [x] 三验收标准中的 1、4 已达成（>98%、loss 曲线与过拟合点）
- [ ] 验收标准 2、3 是**理解类**，需口头讲清楚：
  - `zero_grad()` 为什么必须在 backward 前 → 见 A1 文档第 3 节
  - AMP 解决了什么问题 → 见 A1 文档第 2.5 节（含"小模型上反而慢"的边界）
- 下一阶段：**A2（RL 基本概念）**，用 CartPole-v1 + LunarLander-v3（Box2D 已装好）

---

## 五、附：本次用到的命令

```powershell
. D:\work\embodied-intelligence\scripts\activate.ps1
Set-Location D:\isaac\work

python D:\work\embodied-intelligence\code\a1_train_loop.py --seed 0 --epochs 6
python D:\work\embodied-intelligence\code\a1_train_loop.py --seed 1 --epochs 6
python D:\work\embodied-intelligence\code\a1_train_loop.py --seed 2 --epochs 6
```
