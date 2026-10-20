# Isaac Sim 6.2 安装与 GPU 兼容性验证记录

> 日期：2026-10-20
> 结论：**RTX 5070 Ti（Blackwell sm_120）可正常运行 Isaac Sim 6.2**，C 模块（Isaac Sim / Isaac Lab）可以在本机执行，无需改用云 GPU。
> 关联文档：《具身智能学习与实践计划.md》模块 C

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| Isaac Sim 版本 | 6.2.0.0（pip 安装） |
| GPU 识别 | ✅ `NVIDIA GeForce RTX 5070 Ti`，15995 MB，**Active: Yes** |
| 架构检测 | ✅ `sm_120`（Blackwell），Warp 已识别 |
| 渲染器 | ✅ Vulkan，driver 591.86 |
| Kernel 版本 | `110.4.0+feature.386251.187cf21b.gl` |
| SimulationApp 启动 | ✅ 42.8 秒 |
| 渲染循环（10 帧 update） | ✅ 通过 |
| torch CUDA | ✅ 2.11.0+cu128，cuda.is_available()=True |
| 算力验证 | ✅ fp16 4096³ matmul，21.8 TFLOPS |
| 退出码 | 0 |

**这解决了计划里最大的不确定性。** 原本记录的风险是"50 系兼容性未验证，可能需要云 GPU"，现已实测可用。

---

## 2. 安装位置与体积

| 项 | 路径 | 说明 |
|---|---|---|
| Python 环境 | `D:\isaac\env` | conda 环境（python 3.12.15） |
| site-packages | `D:\isaac\env\Lib\site-packages` | 装着全部 isaacsim_* 包 |
| 环境总大小 | **14.7 GB** | |
| 启动程序 | `D:\isaac\env\Scripts\isaacsim.exe` | 不在 PATH 上，需用全路径 |
| pip 缓存 | `D:\isaac\pipcache` | 可随时删除 |
| 临时目录 | `D:\isaac\temp` | 安装时用，平时可删 |
| 安装日志 | `D:\isaac\install.log` | 排查用 |

**为什么装在 D 盘**：C 盘当时只剩 6.9 GB，而 Isaac Sim 要求 50 GB 以上。D 盘 368 GB。**这是硬性前提，不要再往 C 盘装。**

---

## 3. 安装过程中遇到的三个问题

### 3.1 问题一：C 盘空间不足（`No space left on device`）

**现象**：下载 5.5 GB 的 `isaacsim_extscache_kit` 到 4.2 GB 时失败。

**根因**：pip 安装大包时，下载的临时文件和解压后的临时目录都写在 `%TEMP%`，而 `%TEMP%` 默认是 `C:\Users\ZQM\AppData\Local\Temp`（C 盘只剩 6.9 GB）。**`PIP_CACHE_DIR` 只控制"缓存"，不控制"临时文件"，这两个是不同目录。**

**解法**：

```powershell
New-Item -ItemType Directory -Force -Path D:\isaac\temp | Out-Null
$env:TEMP = "D:\isaac\temp"
$env:TMP = "D:\isaac\temp"
$env:PIP_CACHE_DIR = "D:\isaac\pipcache"
```

**要点**：`$env:` 设置的变量只在当前 PowerShell 窗口有效，换窗口必须重设。

### 3.2 问题二：pip 26 的 PEP 658 严格校验（`BadMetadata: invalid metadata entry 'name'`）

**现象**：依赖解析阶段直接崩溃，报错栈指向 `_check_sidecar_matches_wheel`。

**根因**：pip 26 新增了严格的 PEP 658 校验，会拿索引提供的 `.metadata` 附属文件与 wheel 实际元数据逐字段比对，遇到解析失败的元数据就抛 `BadMetadata`。这是 pip 版本与包格式的兼容问题，不是网络或磁盘问题。

**排查过程**（记录下来避免以后重复）：
1. 检查 `site-packages` 里的 `dist-info` —— 全部正常
2. 检查 pip 内部视角的全部分发包 —— 只有 4 个，全部正常
3. 检查 `sys.path` 上所有分发包 —— 损坏数量 0
4. 检查 pip 缓存里的 76 个 metadata 文件 —— 全部正常
5. 检查 `pypi.nvidia.com` 索引页 —— **NVIDIA 所有子包都不声明 `data-dist-info-metadata`**
6. 读 pip 源码确认 —— `_check_sidecar_matches_wheel` 是 pip 26 新增

**解法**：降级 pip 绕过该校验。

```powershell
D:\isaac\env\python.exe -m pip install "pip<26"
```

降级到 **pip 25.3** 后安装一次成功。已验证 25.3 的 `prepare.py` 里不含 `_check_sidecar_matches_wheel`。

**注意**：不要再无脑升级 pip，升到 26 会重现这个问题。

### 3.3 问题三：torch 是 CPU 版

**现象**：`torch 2.11.0+cpu`，`torch.cuda.is_available()` 返回 `False`。

**根因**：`isaacsim-core` 依赖 `torch==2.11.0`，而 Windows 上 PyPI 提供的 torch 是 **CPU-only 构建**。Isaac Sim 本身用 CUDA 做渲染和物理，不依赖 torch 的 CUDA；但 **Isaac Lab 的强化学习训练需要 GPU 版 torch**，否则策略网络训练会退回 CPU，速度慢几十倍。

**解法**：从 PyTorch 官方 CUDA 源重装。

```powershell
D:\isaac\env\python.exe -m pip install `
    --index-url https://download.pytorch.org/whl/cu128 `
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
```

**为什么选 cu128**：Blackwell（sm_120）需要 CUDA 12.8 及以上。cu128 与 Isaac Sim 自带的 CUDA 12.9 运行时最接近，兼容性风险最低。cu130 也可用，但混用风险略高。

**踩的坑**：第一次直接执行 `pip install torch==2.11.0`，pip 输出全是 `Requirement already satisfied`，**什么都没装**。因为 `2.11.0+cpu` 已经满足 `==2.11.0` 约束（PEP 440 的本地版本号规则）。

必须显式指定带 `+cu128` 的版本，并强制重装：

```powershell
D:\isaac\env\python.exe -m pip install `
    --index-url https://download.pytorch.org/whl/cu128 `
    --force-reinstall --no-deps `
    "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" "torchaudio==2.11.0+cu128"
```

- `--force-reinstall`：覆盖已装的 CPU 版
- `--no-deps`：依赖已满足，避免误动 Isaac Sim 的其他包

**最终验证结果**：

```
torch      : 2.11.0+cu128
cuda avail : True
cuda ver   : 12.8
cudnn      : 91900
device     : NVIDIA GeForce RTX 5070 Ti
capability : (12, 0) (sm_120)
arch list  : ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
matmul 4096^3 fp16: 6.31 ms, 21.8 TFLOPS
```

`arch list` 里包含 `sm_120`，说明这个 torch 构建原生支持 Blackwell，不是靠兼容模式跑。

---

## 4. 验证证据（原始输出摘录）

### 4.1 GPU 识别

```
| Driver Version: 591.86 | Graphics API: Vulkan |
| 0 | NVIDIA GeForce RTX 5070 Ti | Yes: 0 | | 15995 MB | 10de | d21b0100.. |
| 1 | AMD Radeon(TM) Graphics    |        | | 16267 MB | 1002 | f12a0100.. |

| OS: Windows 11 Pro, Version: 10.0 (23H2), Build: 22631
| Processor: AMD Ryzen 5 9600X 6-Core Processor
| Cores: 6 | Logical Cores: 12
| Total Memory (MB): 48291 | Free Memory: 12543
```

### 4.2 Warp / CUDA

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.1
   Devices:
     "cpu"      : "AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD"
     "cuda:0"   : "NVIDIA GeForce RTX 5070 Ti" (16 GiB, sm_120, mempool enabled)
   Kernel cache: C:\Users\ZQM\AppData\Local\NVIDIA\warp\Cache\1.17.0
```

**`sm_120` 是关键证据** —— 这是 Blackwell 的计算能力代号，说明没有降级到兼容模式。

### 4.3 启动与渲染

```
[42.839s] Simulation App Startup Complete
[2] SimulationApp 启动成功, 耗时 45.3s
[3] omni.usd OK
[4] Kit 版本: 110.4.0+feature.386251.187cf21b.gl
[5] Stage 存在: True
[6] 10 帧 update 成功 -> 渲染循环可用
[43.311s] Simulation App Shutting Down
=== 退出码: 0 ===
```

---

## 5. 已知限制与注意事项

### 5.1 无害警告

| 警告 | 是否要处理 |
|---|---|
| `Skipping unsupported non-NVIDIA GPU: AMD Radeon(TM) Graphics` | 不用。这是核显，Isaac 只用独显 |
| `Possible version incompatibility. Attempting to load carb::ujitso::IFactory with version v0.3 against v0.2` | 不用。扩展版本号提示，不影响功能 |
| `pxr.Semantics is deprecated` | 不用。SDK 内部弃用提示 |
| `Core::getServerInfo: rejecting empty URL as invalid` | 不用。没配 Nucleus 服务器，本地模式正常 |

**没有出现**：降噪器失效、渲染噪点、CUDA 报错 —— 说明 6.2 版本对 50 系的支持已经完善（4.5 时代的降噪器问题已修复）。

### 5.2 内存偏紧

验证时 `Free Memory: 12543 MB`。48 GB 物理内存里，WSL 占了一部分，Isaac Sim 启动又吃掉十几 GB。

**建议**：运行 Isaac Sim 前关掉 WSL 里的大型进程（Gazebo、RViz），或考虑调整 `.wslconfig` 把 WSL 内存限制到 16 GB 以下，给 Windows 侧留空间。

### 5.3 EULA 必须接受

非交互环境下必须设环境变量，否则 `import isaacsim` 会卡在询问：

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

**这个变量只在当前窗口有效**，每次新开窗口都要设。

### 5.4 版本锁定（不要随意升级）

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.15 | Isaac Sim 6.x 强制要求 3.12 |
| pip | **25.3** | 升到 26 会重现 BadMetadata |
| Isaac Sim | 6.2.0.0 | |
| torch | 2.11.0+cu128 | |
| mujoco | 3.12.0 | 被 isaacsim-core 一并装上 |
| mujoco-warp | 3.12.0 | 同上，可用于 GPU 并行 RL |
| newton | 1.6.1 | 同上 |

**意外收获**：`isaacsim[all,extscache]` 顺带装了 **mujoco 3.12.0 + mujoco-warp + newton**。也就是说模块 B（MuJoCo）所需的东西已经在同一个环境里了，可以直接用。

---

## 6. 标准启动方式

每次启动 Isaac Sim 前，在 PowerShell 里执行：

```powershell
conda activate D:\isaac\env
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:TEMP = "D:\isaac\temp"
$env:TMP = "D:\isaac\temp"

# 图形界面
D:\isaac\env\Scripts\isaacsim.exe

# 或者 headless（训练用）
D:\isaac\env\python.exe your_script.py
```

**每一步的作用**：
- `conda activate` —— 切到装了 Isaac Sim 的环境（不切会用 base 环境，找不到包）
- `OMNI_KIT_ACCEPT_EULA` —— 跳过许可证交互询问
- `TEMP`/`TMP` —— 防止 Isaac 的缓存和日志往 C 盘写

---

## 7. 对学习计划的影响

### 7.1 计划变更

| 原计划内容 | 变更 |
|---|---|
| "C 模块可能需要云 GPU 或纯 MuJoCo 兜底" | **取消**。本地可跑，按原计划执行 |
| "第一周先验证 Isaac Sim 能否跑" | **已完成**。第一周可直接进入模块 A |

### 7.2 可以立即开始的事

1. **模块 A**：PyTorch 与 RL 基础（环境已就绪）
2. **模块 B**：MuJoCo（已随 Isaac Sim 装上 3.12.0）
3. **模块 C**：Isaac Lab 安装（下一步）

### 7.3 下一步建议

**安装 Isaac Lab**。它是建立在 Isaac Sim 之上的 RL 训练框架，提供并行环境和任务定义。Isaac Lab 2.x/3.x 支持对接已有的 Isaac Sim 安装。

需要注意：Isaac Lab 对 Isaac Sim 版本有配对要求，安装前先查兼容性表。

---

*记录人：AI 助手*
*验证时间：2026-10-20 19:41*
*环境：Windows 11 23H2 / RTX 5070 Ti 16GB / Ryzen 5 9600X / 48GB RAM*