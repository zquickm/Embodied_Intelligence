# code 目录

> 创建：2026-10-21
> 用途：存放**学习用的手写代码**——训练循环、RL 算法实现、世界模型组件。

## 为什么单独建这个目录

原先所有代码散在 `ros2_ws/`（ROS 2 包）和根目录（`gpu_test.py`）。从 A1 开始会产生一批**独立于 ROS/Isaac 的纯 PyTorch 代码**，它们：

1. 不属于任何 ROS 2 包，放 `ros2_ws/` 里会被 colcon 误扫
2. 不属于环境验证脚本，放根目录会让根目录变乱
3. **是求职时最该展示的资产**——"我能手写 PPO/SAC/RSSM"靠这些文件证明

## 目录规划

```
code/
├── a1_train_loop.py        ← A1：手写训练循环（MNIST）
├── a2_*.py                 ← A2：RL 基本概念（策略梯度等）
├── a3_ppo.py               ← A3：手写 PPO
├── a4_sac.py               ← A4：手写 SAC
├── a5_*.py                 ← A5：多种子实验工具
├── b*.py                   ← B 模块：MuJoCo 相关
└── d*/                     ← D 模块：世界模型（VAE / RSSM / Dreamer）
```

**命名规则**：`<阶段号>_<内容>.py`，小写下划线。这样一看文件名就知道对应哪个阶段，和 `file/` 里的记录文档能对上。

## 两条约定

1. **每个文件顶部必须有 docstring**，写清：这段代码做什么、对应哪个阶段、怎么运行。
2. **运行产物不写在这里**。日志、checkpoint、图 → `D:\isaac\work`；最终要展示的曲线图 → `file/assets/`。

## 与训练目录的关系

```
代码在仓库：  D:\work\embodied-intelligence\code\a1_train_loop.py
运行在别处：  Set-Location D:\isaac\work
              python D:\work\embodied-intelligence\code\a1_train_loop.py
产物在这里：  D:\isaac\work\logs\...
```

**为什么不在仓库目录里跑**：日志会写进仓库，污染 git（C1 阶段踩过这个坑，见《阶段C1详细操作步骤.md》7.2）。
