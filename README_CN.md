# arc-llama

> 面向 Intel Arc GPU 的开箱即用 `llama.cpp` 运行时。

`arc-llama` 是一个命令行工具，它能自动检测你的 Intel Arc 显卡，按代际匹配正确的 SYCL/oneAPI 环境，下载或注册 GGUF 模型，并在它们前面跑起一个 OpenAI 兼容的服务。它把那些坑（持久设备代码缓存里的 SIGSEGV、IPEX-LLM 的环境变量陷阱、每代架构的 KV-cache 量化行为）都封装起来，省得你亲自踩一遍。

它的目标就是：你拆开 Arc 显卡、装好驱动，上午就能用上本地大模型。

> [!NOTE]
> **版本：0.6.0。** 在 Battlemage B60 上完整跑通：`arc-llama install-runtime`
> 会拉取一个便携版 Vulkan `llama-server`，无需安装 oneAPI、无需源码编译即可提供真实推理。
> HF 下载、流式响应、OpenAI 兼容 API 均已通过。其他 SKU（A770、A380、B580）还需要社区验证——
> 如果你的卡跑不通，请开 issue。

## 你能得到什么

- **自动发现 GPU 和模型。** `arc-llama init` 会找到你的 Intel 显卡，并按配置的扫描路径查找 `.gguf` 文件，把每个模型都注册进来，上下文长度按你的显存自动裁剪，KV-cache 类型按文件名推断。已经放在磁盘上的 GGUF，你基本不需要手动 `arc-llama add`。
- **自动发现**主机上的每一块 Intel GPU（`Alchemist`、`Battlemage`、Lunar Lake 核显）。PCI 设备 ID 表覆盖常见 SKU，其余型号会回退到 OpenCL 设备名解析。
- **按架构的 SYCL 配置文件**，`SYCL_CACHE_PERSISTENT=0` 等环境变量会自动应用；已知有害的变量（如 `GGML_SYCL_DISABLE_OPT`、`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`）会从继承的 shell 环境中被剥离。
- **智能默认值**，根据检测到的显存和模型的量化张量大小自动设置 `-ctx`、`--cache-type-k/v`、`-ngl`，绝不会启动一个你显存放不下的模型。
- **TOML 模型注册表**，位于 `$XDG_CONFIG_HOME/arc-llama/config.toml`，随手可改。
- **每个模型一个进程**，由内部路由器按需换入换出。默认策略是跨所有 GPU 单驻留（有利于散热）；显存充裕时可以改成多驻留。
- **OpenAI 兼容 API**，`http://127.0.0.1:11437/v1/...`。可接入 Open WebUI、OpenCode、任何支持 OpenAI API 的工具。
- **内置 Web UI**，`http://127.0.0.1:11437/`。支持模型选择、加载/停止按钮、**在线修改 ctx 和 KV 量化**、GPU/显存面板。纯 HTML/JS，无需构建。
- **终端 UI**（`arc-llama tui`），基于 Textual，拥有与 Web UI 相同的加载/停止/编辑控制。可选安装：`pip install 'arc-llama[tui]'`。
- **后台自动调优。** 丢进一个 GGUF，使用一次后，`arc-llama serve` 会在下一个空闲窗口测量出更快的配方——无需手动 tune。扫描在有真实请求到达时会立即中止。
- **不绑架你现有的栈。** 它调用你自己的 `llama-server` 二进制文件，你不会被锁定在某个特定版本。

## 快速开始

```bash
# 1. 安装
pip install arc-llama

# 或者开发模式安装：
# git clone https://github.com/offbyonebit/arc-llama
# cd arc-llama
# pip install -e .

# 2. 检测 GPU 并写入初始配置（此时还不需要 llama-server）
arc-llama init

# 3. 下载便携版 Vulkan llama-server 并写入配置。
#    无需 oneAPI、无需编译 llama.cpp。（如需 SYCL 构建，加 --backend sycl）
arc-llama install-runtime

# 4. 查看当前状态
arc-llama doctor
arc-llama gpus

# 5. 自动注册扫描路径下的所有 GGUF。
#    `init` 已经跑过一次；新文件丢进去后可随时重跑。
arc-llama scan
# （一次性添加：arc-llama add /path/to/some.gguf）
# （从 Hugging Face：arc-llama add unsloth/gemma-4-31B-it-GGUF:Q4_K_M --from-hf）

# 6. 启动 OpenAI 兼容服务（同时提供 Web UI）
arc-llama serve

# 7. 丢一个 GGUF 进去，用一次 —— 空闲后自动调优会触发，
#    或者现在就手动调优：
arc-llama benchmark <model>
arc-llama tune <model>
arc-llama tune --status            # 仅打印调优状态，不扫描
arc-llama serve --no-auto-tune     # 禁用后台扫描

# 8. （可选）在另一个窗口打开终端 UI
arc-llama tui

# 9. （可选）写入 systemd --user 单元
arc-llama systemd --write
systemctl --user daemon-reload
systemctl --user enable --now arc-llama.service
```

然后让任意 OpenAI 兼容客户端指向 `http://127.0.0.1:11437/v1`：

```bash
curl http://127.0.0.1:11437/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-q4_k_m",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

## 系统要求

- Linux，Battlemage 推荐内核 **6.14+**（`xe` 驱动；6.8 是 `xe` 存在的最低版本，但 BMG 在 6.14+ 才稳定），Alchemist 推荐 5.17+（`i915`）。`arc-llama doctor` 会提示这个阈值。
- BIOS 中启用 ReBAR，否则 llama.cpp 在 Arc 上会走慢路径。
- 需一个使用 SYCL 后端构建的 `llama-server`。Intel oneAPI Base Toolkit 是支持的构建路径：
  ```bash
  source /opt/intel/oneapi/setvars.sh
  cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
  cmake --build build --config Release -j
  ```
- 用户需在 `render` 和 `video` 组（`arc-llama doctor` 会提示）。

## 基准测试与自动调优

静态默认值无法知道*你的*卡/模型/llama.cpp 构建更喜欢 f16 还是 q8_0 KV-cache、512 还是 2048 的 ubatch、flash attention 开不开——SYCL 后端的真实答案因 SKU 和版本而异。所以直接测量：

```bash
# 一次性测量（prompt-eval + 生成 tok/s、显存占用）
arc-llama benchmark qwen3-7b

# 扫描上下文长度 × KV 类型
arc-llama benchmark qwen3-7b --sweep-ctx 8192,32768 --kv f16 --kv q8_0

# 分阶段贪婪扫描：KV 类型 → ubatch → flash attention。获胜者写入模型 recipe 并持久化。
# Battlemage 上约 6–9 组配置、约 10 分钟。
arc-llama tune qwen3-7b
arc-llama tune qwen3-7b --target generation   # 只优化聊天延迟
arc-llama tune qwen3-7b --dry-run             # 只看不动
arc-llama tune --status                       # 打印状态，不扫描
```

手动 `tune` 需要 `arc-llama serve` 正在运行，这样测量才能继承真实请求会用到的 SYCL 环境和路由器策略。启用后，后台自动调优通过本地回环 HTTP 跑同样的 `tune_model` 路径。设置 `[tune] auto = false` 或传 `--no-auto-tune` 可禁用。候选配置启动失败（例如更大的 ubatch 导致 compute-buffer OOM）会输掉该轮——调优器始终保证模型以可用配置运行。

`arc-llama` 还会针对每个 `llama-server` 二进制文件探测一次 `--help`，从而输出正确版本的 flag 方言（当前构建使用 `-fa on|off|auto`，b6300 之前的构建使用布尔 `-fa`），所以手工构建和预构建二进制都能用。

## 多 GPU

`arc-llama init` 会注册它找到的每一块 Intel GPU。配置中的每个模型都绑定到具体 PCI 插槽，SYCL 设备选择器（`ONEAPI_DEVICE_SELECTOR=level_zero:N`）按模型设置。加装第二块卡后，运行 `arc-llama init --force` 刷新 `[[gpus]]`，然后可以往任意 GPU 上添加模型。

默认换入策略是**跨所有 GPU 单驻留**：选中一个模型时，路由器会先停止其他模型。在配置中把 `server.single_resident` 设为 `false` 可让不同 GPU 上的模型共存。

## 上游端点

`arc-llama` 可以把其他 OpenAI 兼容端点（如 Ollama、vLLM、另一个 arc-llama 实例）的模型合并进自己的模型列表，并透明地转发请求：

```bash
# 添加上游
arc-llama upstream add ollama http://127.0.0.1:11434

# 列出上游
arc-llama upstream list

# 删除
arc-llama upstream remove ollama
```

上游模型会出现在 `/v1/models` 中，`owned_by: "upstream:NAME"`，请求直接路由到上游端点，本地不启动 llama-server。模型列表缓存 30 秒，按需刷新。

## 配置参考

`$XDG_CONFIG_HOME/arc-llama/config.toml`：

```toml
version = 1

[server]
host = "127.0.0.1"
port = 11437
single_resident = true

[paths]
llama_server = "/usr/local/bin/llama-server"
models_dir   = "~/.local/share/arc-llama/models"
state_dir    = "~/.local/state/arc-llama"

[tune]
auto         = true      # 空闲时后台扫描
idle_seconds = 120

[[gpus]]
pci_slot   = "0000:03:00.0"
sycl_index = 0
arch       = "battlemage"
backend    = "sycl"          # 若使用 Vulkan llama-server 构建，可改为 "vulkan"
vram_mb    = 24480
enabled    = true
name       = "Arc Pro B60"

[[models]]
name             = "qwen3-7b"
display_name     = "Qwen 3 7B"
path             = "/home/me/models/qwen3-7b-q4_k_m.gguf"
gpu_pci_slot     = "0000:03:00.0"
port             = 18080
kv_class         = "default"
aliases          = ["qwen3-7b-q4_k_m.gguf"]

[models.recipe]
ctx              = 32768
cache_type_k     = "q8_0"
cache_type_v     = "q8_0"
n_gpu_layers     = 999
parallel         = 1
extra_flags      = []

[[upstreams]]
name = "ollama"
url  = "http://127.0.0.1:11434"
```

> [!NOTE]
> 可选的 agent/编程助手模式为实验性。在运行 `arc-llama agent`、`code` 或 `agent-tui` 之前，先设置环境变量 `ARC_LLAMA_EXPERIMENTAL_AGENT=1`。

`kv_class` 控制 `arc-llama add` 估算 KV-cache 大小时使用的每 token 占用。当前选项：

| 值                | 每 token f16 KV | 典型场景                                      |
|-------------------|-----------------|-----------------------------------------------|
| `default`         | ~80 KiB         | 多数 ≤30B 稠密模型，保守上限                  |
| `qwen3_27b_dense` | ~70 KiB         | Qwen 3 27B 稠密                               |
| `moe_a3b`         | ~24 KiB         | Qwen 3 30B/35B-A3B MoE                        |
| `gemma_swa`       | ~16 KiB         | Gemma 3/4（交错滑动窗口注意力）               |

## 架构

```
┌──────────────────────┐
│  OpenAI 客户端       │  Open WebUI、OpenCode、curl …
│  (端口 11437)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  arc-llama serve     │  FastAPI，/v1/chat/completions 等
│  (路由器 + 状态)     │
└──────────┬───────────┘
           │ ensure_active(model)
           ▼
┌──────────────────────┐
│  Router              │  按请求换入 llama-server 子进程
│  (单/多驻留)         │  应用架构 SYCL 环境、选择安全的 ctx/KV
└──────────┬───────────┘
           │ subprocess.Popen
           ▼
┌──────────────────────┐
│  llama-server (SYCL) │  每个注册模型按需一个实例，绑定到 GPU N
└──────────────────────┘
```

路由器使用 `asyncio.Lock` 序列化换入操作，因此对同一模型的并发请求会共享一个已预热后端。健康检查轮询 `{backend_url}/health`；冷启动预算默认 120 秒，以吸收普通 `llama.cpp` 每次新启动都要支付的 SYCL JIT 重编译时间。

## 为什么不直接用 Ollama / vLLM？

- **Ollama（IPEX-LLM 封装）：** Intel 支持的分支在 Battlemage 上运行 Qwen2.5 类模型时有可复现的推理 bug，连续调用会退化成 NaN 导致的乱码。`arc-llama` 直接运行 `llama-server`，完全避开那条路径。
- **vLLM-XPU：** 在 Arc 上仍在成熟中，量化支持较弱。如果你要跑稠密 >30B 且追求吞吐，可以一试，但还不是一键体验。
- **裸 `llama-server` + 脚本：** 这正是大多数 Arc 用户现在的做法。`arc-llama` 是这些脚本的正式化，把坑都提前埋好。

## UI

项目内置两个前端，都调用同一套 admin 端点（`/admin/status`、`/admin/load/{name}`、`/admin/stop/{name}`、`/admin/stop-all`）：

- **Web UI**：`http://<host>:<port>/`（默认 `127.0.0.1:11437`）。静态页面每 5 秒刷新。显示状态、GPU、模型列表、每个模型的 Load/Stop 按钮、“Stop all” 紧急按钮。无需构建，无 JS 依赖。
- **Terminal UI**：通过 `arc-llama tui` 启动，基于 Textual。快捷键：`r` 刷新、`l` 加载选中模型、`s` 停止选中、`S` 全部停止、`q` 退出。可与 `arc-llama serve` 同时运行（或针对远程服务传 `--server`）。

两者都使用亮度/暗淡来表示状态（加载/空闲），不使用红绿配色。

## 容器

项目包含 Dockerfile，可构建带 SYCL 后端（默认开启 FP16 数学路径）的 `llama-server` 并单镜像安装 `arc-llama`：

```bash
# 构建（通用：JIT 编译设备代码，适用于任意 Intel GPU）
docker build -t arc-llama:latest .

# 为你的 GPU 代际预编译 AOT 设备代码 —— 消除每次冷启动约 20 秒的 SYCL JIT 重编译
# （Battlemage 无法使用 JIT 缓存）：
docker build --build-arg GGML_SYCL_DEVICE_ARCH=bmg-g21 -t arc-llama:bmg .  # B 系列
docker build --build-arg GGML_SYCL_DEVICE_ARCH=acm-g10 -t arc-llama:acm .  # A770/750/580

# 运行（需要 GPU 访问）
docker run --rm -it \
  --device /dev/dri:/dev/dri \
  --group-add video --group-add render \
  -p 11437:11437 \
  -v $HOME/models:/models:ro \
  arc-llama:latest
```

Entrypoint 会在首次启动且无配置时自动运行 `arc-llama init`，然后启动 `arc-llama serve`。如需完全控制，可挂载你自己的 `config.toml`：

```bash
docker run ... \
  -v $PWD/config.toml:/root/.config/arc-llama/config.toml:ro \
  arc-llama:latest
```

## 路线图

- ~~HF 模型下载（`arc-llama add org/repo:quant --from-hf`）。~~ ✅
- ~~流式响应转发（`stream: true`）。~~ ✅
- ~~预构建 `llama-server` + arc-llama 容器镜像。~~ ✅
- ~~`arc-llama benchmark` 快速 prompt-eval/生成 tok/s 测试。~~ ✅
- ~~`arc-llama tune` 测量并持久化配方的自动调优器。~~ ✅
- ~~`arc-llama install-runtime` 下载预构建 llama-server（默认 Vulkan）。~~ ✅
- ~~`arc-llama tune --all` 一键扫描所有已注册模型。~~ ✅
- ~~首次使用后后台自动调优，新请求到达时中止。~~ ✅

## 贡献

欢迎 PR 和 issue。当前最有价值的贡献：

1. 确认或修正你显卡对应的 PCI 设备 ID → 架构映射。如果 `arc-llama gpus` 把一块能正常工作的 Arc 卡显示为 `unknown`，请带 `lspci -nn` 输出开 issue。
2. 报告默认 SYCL 环境配置会崩溃或性能低下的架构。
3. 在维护者以外的硬件（目前主要是 Battlemage B60 开发机）上跑 smoke test。

## 支持

本项目免费，不索取任何回报。如果对你有用，给仓库点个 star 就好；如果还想关注我别的项目，可以在 GitHub 找到 [@offbyonebit](https://github.com/offbyonebit)。

如果你想赞助开发，可以 [在 GitHub 赞助我](https://github.com/sponsors/offbyonebit)。

## 许可

MIT，详见 [LICENSE](LICENSE)。
