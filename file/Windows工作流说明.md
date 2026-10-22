# Windows 工作流说明

> 生效日期：2026-10-21
> 背景：核心模块（C Isaac Sim/Lab、D 世界模型）都在 Windows 侧，仓库已从 WSL 迁到 Windows。

---

## 1. 三个关键位置

| 用途 | 路径 | 说明 |
|---|---|---|
| **代码仓库** | `D:\work\embodied-intelligence` | git 仓库，文档 + 代码的唯一真实来源 |
| **Python 环境** | `D:\isaac\env` | conda 环境，装了 Isaac Sim 6.2 + Isaac Lab 3.0 + torch cu128 |
| **实验产物** | `D:\isaac\work` | 训练日志、checkpoint，**不进 git** |

**三者分开的意义**：仓库保持干净（只有代码和文档），环境可独立重建，实验产物可随时清空不影响代码。

---

## 2. 每次开窗先激活

```powershell
. D:\work\embodied-intelligence\scripts\activate.ps1
```

**注意 `=` 前面有一个点和一个空格**。点号表示"在当前作用域执行"，否则设的环境变量不会留在当前窗口。

**这个脚本做了 7 件事**：

| # | 动作 | 不做的后果 |
|---|---|---|
| 1 | 设 `OMNI_KIT_ACCEPT_EULA=YES` | `import isaacsim` 卡在等待输入 |
| 2 | `TEMP`/`TMP` 指向 D 盘 | C 盘只剩几 GB，pip 装大包必爆 |
| 3 | `PIP_CACHE_DIR` 指向 D 盘 | 缓存占 C 盘 |
| 4 | `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8` | **中文控制台 GBK 编码崩溃**（见第 5 节） |
| 5 | 激活 conda 环境 `D:\isaac\env` | 用成 base 环境，找不到 isaaclab |
| 6 | `D:\isaac\env\Scripts` 加入 PATH | 要敲全路径才能调 `isaaclab` |
| 7 | 回显 python 路径、版本、CUDA 状态 | 环境错了也不知道 |

**激活成功的标志**：

```
================ Isaac env ready ================
  python  : D:\isaac\env\python.exe
  version : 3.12.15
  torch   : 2.11.0+cu128 | cuda: True
  TEMP    : D:\isaac\temp
  repo    : D:\work\embodied-intelligence
=================================================
```

**看到 `cuda: True` 才算对**。如果是 `False`，说明装成了 CPU 版 torch。

---

## 3. 常用命令

### 3.1 看有哪些任务

```powershell
isaaclab list_envs --all
isaaclab list_envs --keyword Franka
```

### 3.2 训练

```powershell
# 建议在独立目录跑，避免日志写进仓库
Set-Location D:\isaac\work
isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --num_envs 4096 --max_iterations 150
```

### 3.3 回放训练好的策略

```powershell
isaaclab play --rl_library rsl_rl --task Isaac-Cartpole `
    --checkpoint "D:\isaac\work\logs\rsl_rl\cartpole\<时间戳>\model_149.pt" `
    --num_envs 16 --real-time
```

`play` 会开窗口并一直运行，看完手动关。

### 3.4 看训练曲线

```powershell
tensorboard --logdir D:\isaac\work\logs
```

然后浏览器打开 `http://localhost:6006`。

**checkpoint 位置规律**：

```
D:\isaac\work\logs\rsl_rl\<experiment_name>\<时间戳>\model_<迭代号>.pt
```

---

## 4. 目录约定

```
D:\
├── isaac\
│   ├── env\            ← Python 环境（20 GB，不要手动改）
│   ├── work\           ← 训练实验产物（可清空）
│   │   ├── logs\       ← TensorBoard + checkpoint
│   │   ├── data\       ← 数据集（MNIST 等，2026-10-21 新增）
│   │   └── *.log       ← 训练日志
│   ├── temp\           ← 临时文件（可清空）
│   └── pipcache\       ← pip 缓存（可清空）
└── work\
    └── embodied-intelligence\   ← git 仓库
        ├── file\        ← 所有文档
        │   └── assets\  ← 训练曲线图、数据表（进 git，2026-10-21 新增）
        ├── code\        ← 学习代码（进 git，2026-10-21 新增）
        ├── scripts\     ← activate.ps1 等工具脚本
        ├── ros2_ws\     ← ROS 2 包（WSL 相关，暂时不用）
        └── gazebo_media\ ← Gazebo 资源（同上）
```

**新增目录的原因**：A1 起会产出两类新东西，原先的约定没覆盖。

| 目录 | 放什么 | 为什么不放别处 |
|---|---|---|
| `code\` | 手写的训练循环、RL 算法实现（A1–A5、D1–D3） | 这些是"能讲解"的核心资产，必须进 git；放 `D:\isaac\work` 会随清空丢失 |
| `file\assets\` | 训练曲线 png、超参表 csv、多种子汇总表 | 属于"派生产物"，体积小、可复现，进 git 后文档里的图才有出处 |
| `D:\isaac\work\data\` | MNIST 等数据集 | 体积大、可重新下载，**不进 git** |

**判断口诀**：**图、表、模板、代码 → 进仓库；原始日志、checkpoint、数据集 → 留在 isaac 盘。**

---

## 4.1 pip 装包的安全流程（2026-10-21 立规）

Isaac 环境的依赖 pin 很脆（见第 5 节）。**任何 `pip install` 前必须先 dry-run**：

```powershell
# 第 1 步：先看会发生什么（不实际安装）
python -m pip install --dry-run "包名==版本"

# 第 2 步：确认输出里只有 "Would install"，没有版本变更
# 第 3 步：确认无误再执行，前后各存一次快照
python -m pip freeze | Out-File D:\isaac\work\pipfreeze_before.txt -Encoding UTF8
python -m pip install "包名==版本"
python -m pip freeze | Out-File D:\isaac\work\pipfreeze_after.txt -Encoding UTF8
Compare-Object (Get-Content D:\isaac\work\pipfreeze_before.txt) (Get-Content D:\isaac\work\pipfreeze_after.txt)
```

**判定标准**：

- 只有 `=>`（新增）→ 安全
- 出现 `<=`（删除/版本变更）→ **停下**，先查这个包为什么动了 pin

**装完后必须回归**：跑一次 `isaaclab train --task Isaac-Cartpole --num_envs 256 --max_iterations 20`，确认 Isaac Lab 没被破坏。

**实证案例**：2026-10-21 装 Box2D/SB3/pygame-ce/pandas，dry-run 显示只新增 5 个包，实际执行后 `Compare-Object` 确认 **0 个已有包变更**，回归训练 exit=0。这套流程有效。

---

## 5. 三个必须记住的坑

### 5.1 中文控制台编码

**现象**：训练刚启动就退出，报

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u259b'
```

**原因**：中文 Windows 代码页是 936（GBK），Isaac Lab 启动横幅含制表符 `▛`，GBK 编不出来。

**解决**：`$env:PYTHONUTF8 = "1"`（activate.ps1 已包含）。

### 5.2 .ps1 文件必须 UTF-8 with BOM

**现象**：脚本报一堆 `表达式或语句中包含意外的标记` 解析错误，中文注释变乱码。

**原因**：Windows PowerShell 5.1 读取**无 BOM** 的 .ps1 时用系统代码页（GBK）解码，UTF-8 中文被解错，字节还可能破坏引号配对。

**解决**：所有含中文的 `.ps1` 必须保存为 **UTF-8 with BOM**（前 3 字节 `EF BB BF`）。

**注意**：`.md` 文件不需要 BOM，保持 UTF-8 无 BOM 即可。

### 5.3 不要升级 pip

当前锁定 **pip 25.3**。升到 26 会触发 PEP 658 严格校验，安装 Isaac Sim 时抛 `BadMetadata` 崩溃。

### 5.4 PowerShell 5.1 的 Get-Content 会误读中文

**现象**：用 `Get-Content 中文文件.md` 看内容，中文全变乱码（如 `运行设备` 显示成 `杩愯璁惧`），但文件本身是好的。

**原因**：Windows PowerShell 5.1 的 `Get-Content` **默认按系统代码页（GBK）解码**，即使文件是 UTF-8 也会读错。而 PS 7（pwsh）默认就是 UTF-8，所以这个问题只在 5.1 上出现。

**解决**：读中文文件时显式指定编码：

```powershell
Get-Content file.md -Encoding UTF8
```

**判断文件是否真的坏了**：如果只是 `Get-Content` 显示乱码、但按 UTF-8 读取正常，那是显示问题不是文件问题。用下面这段验证：

```powershell
$b = [System.IO.File]::ReadAllBytes("文件路径")
$s = [System.Text.Encoding]::UTF8.GetString($b)
([regex]::Matches($s, [char]0xFFFD)).Count   # 0 表示文件是好的
```

**注意**：写文件同理，`Set-Content` 在 5.1 默认写 GBK。要写 UTF-8 用：

```powershell
[System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))
```

`$false` 表示不加 BOM（`.md`/`.py` 用这个），`$true` 表示加 BOM（`.ps1` 必须用这个）。

### 5.5 git push 报 schannel SSL/TLS 握手失败

**现象**：

```
fatal: unable to access 'https://github.com/...': schannel: failed to receive handshake, SSL/TLS connection failed
```

奇怪的是 `git fetch` 有时能成功，只有 `git push` 失败。

**根因**：Windows 版 git 默认用 **schannel**（系统 TLS 栈），在特定网络环境（尤其开了代理软件时）会和 GitHub 的 TLS 握手出问题。`push` 的 POST 数据量大，比 `fetch` 更容易触发。

---

#### ✅ 正确解法：换 SSL 后端为 OpenSSL（2026-10-22 实测确认）

```powershell
# 一次性（本次生效）
git -c http.sslBackend=openssl push origin main

# 长期（写进本仓库配置，推荐）
git config http.sslBackend openssl
```

**实测结论（2026-10-22）**：

| 尝试 | 结果 |
|---|---|
| `git push`（默认 schannel） | ❌ 握手失败 |
| `git -c http.proxy=http://127.0.0.1:7897 push` | ❌ 仍然失败 |
| **`git -c http.sslBackend=openssl push`** | ✅ **成功** |
| `git -c http.sslBackend=openssl fetch`（不加代理） | ✅ 成功，且**不需要代理** |

**重要更正**：本文档早前记录的"走本地代理"方案**不完整**。代理端口确实在监听（7897、7890 都可达），但**加代理并不能解决 schannel 的问题**——真正的病根是 TLS 后端，不是网络出不去。

**为什么 openssl 前端没有额外依赖**：Windows 版 git 自带 OpenSSL 库，`http.sslBackend=openssl` 只是切换 git 使用哪一套 TLS 实现，不需要额外安装东西。

**为什么建议写进仓库配置而不是全局**：这是本机 git 的 TLS 栈问题，跟具体网络环境有关，写进局部配置对未来新克隆的仓库不生效（局部配置不随仓库走），但至少当前工作副本不用每次手动加 `-c`。如果换机器或换网络后出现同样症状，再用同样两条命令。

---

#### 备选方案（本次未采用）

```powershell
# 强制 HTTP/1.1，绕过 HTTP/2 相关的握手问题
git -c http.version=HTTP/1.1 push origin main
```

**仍然可以用代理**，但要知道它解决的是另一类问题（出口被墙），不是 schannel：

```powershell
Get-NetTCPConnection -State Listen |
    Where-Object { $_.LocalPort -in @(7890,7897,10808,10809) } |
    Select-Object LocalPort, OwningProcess -Unique

git -c http.proxy=http://127.0.0.1:7897 push origin main
```

**端口说明**：2026-10-22 实测 **7897 和 7890 都在监听**。端口号取决于代理软件版本，变了自己查上一条命令。

**判断顺序**（下次遇到先按这个顺序试，省时间）：

1. `git -c http.sslBackend=openssl push` ← **先试这个**
2. 还不行 → 再考虑代理 `git -c http.proxy=...`
3. 还不行 → `git -c http.version=HTTP/1.1`

---

## 6. WSL 侧的处置

**WSL 里的原仓库保留作为备份**，路径 `/home/venus/projects/embodied-intelligence`，停留在提交 `6762d62`。

**不再在 WSL 里开发。** 如果以后要做 F/G 模块（ROS 2 + Gazebo），有两种选择：

1. 在 WSL 里重新 `git clone` 一份（推荐，避免 /mnt/d 上 colcon 编译慢）
2. 把 `ros2_ws` 单独拷回 WSL 编译

**WSL 环境本身没有删**：ROS 2 Jazzy、Gazebo Harmonic、turtlebot3 都还在。

---

## 7. 迁移记录

| 项 | 内容 |
|---|---|
| 迁移方式 | `git bundle`（因为本地有 13 个未推送提交，不能直接从 GitHub clone） |
| 完整性校验 | 树哈希一致：`c6f162c45b161960922de63174982dcafd465e10` |
| 提交数 | 17（迁移前后一致） |
| 跟踪文件 | 33（一致） |
| 换行符处理 | 原 `core.autocrlf=true` 导致检出为 CRLF，已改为 `false` + `eol=lf`，并加 `.gitattributes` 锁定 |
| 远程 | 已指向 `https://github.com/zquickm/Embodied_Intelligence.git` |

---

*创建：2026-10-21*