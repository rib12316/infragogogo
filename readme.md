# FlagOS 开放计算全球挑战赛 S2 · 赛道二 · 复现说明

> **队伍**：infragogogo（xxinfra）
> **对象模型**：MiniCPM5-2B（`/workspace/MiniCPM5-2B`）
> **硬件**：天数智芯 BI-V150（32 GB）×1；沐曦 C500（64 GB）×1
> **框架**：vllm-plugin-FL `flagos-2026-s2`（基线 commit `13eb9be`）+ FlagGems `v5.3.5`（未改算子库）
> **目标指标**：官方 benchmark 的 `Total tok/s`（4K、16K 双场景），精度门槛 math_500 Level 3 ≥ 0.95

本说明按“环境 → 安装 → 生成量化 ckpt → 启动服务 → 跑官方 benchmark → 跑精度评测 → 校验结果”的顺序给出完整、可落地的复现步骤。所有优化均以**代码级默认开启**的形式交付：官方评测命令除 `--model` 指向由原始模型**幂等再生**的 W8A8 checkpoint 外，与赛题基线命令一字不差。

---

## 0. 交付物结构

```
infragogogo/
├── vllm-plugin-FL/          # 全部框架侧优化源码（Apache-2.0），含量化转换工具
│   └── tools/quantize_int8_rtn.py
├── FlagGems/                # FlagGems v5.3.5 原样副本（用于审计；本作品未改算子库）
├── scripts/
│   ├── quant/common.py      # 量化转换脚本的折叠数学依赖（必须与 vllm-plugin-FL 保持同根）
│   ├── sweep_attn_launch.py # O5 配置表再生工具（可选）
│   ├── micro_bench_attn_launch.py  # attention 微基准（可选）
│   └── probe/               # 文本对拍、采样过滤微基准（可选）
├── evidence/                # 提交用 benchmark 原始结果、精度报告、act_stats 与 provenance
├── readme.md                # 本文件
└── report.pdf               # 技术报告
```

> **路径约束**：`vllm-plugin-FL/tools/quantize_int8_rtn.py` 会以 `vllm-plugin-FL` 的父目录为仓库根，导入 `scripts/quant/common.py`。请保持本包的以上相对目录结构，不要单独移动 `vllm-plugin-FL/`。

---

## 1. 软硬件环境

### 1.1 天数 BI-V150（正式提交数字所在平台）

| 项 | 版本 / 规格 |
|---|---|
| 加速卡 | 天数智芯 BI-V150，单卡 32 GB，16 SM，SMEM 128 KB/SM，L2 16 MB |
| Python | 3.12 |
| PyTorch | 2.10（CoreX 构建） |
| vLLM | 0.24.0+empty（镜像内置） |
| Triton | 3.6.0（flagtree 0.6.2a3 + iluvatar backend） |
| FlagGems | **v5.3.5**（tag `v5.3.5`，commit `a7620cc`） |
| Evalscope | 镜像内置（本作品使用 1.11.x 同接口） |

### 1.2 沐曦 C500（适配与测试平台；正式数字测试中，见 §8）

| 项 | 版本 / 规格 |
|---|---|
| 加速卡 | 沐曦曦云 C500，64 GB，104 SM |
| MACA / 驱动 | MACA 3.7.0.38；driver 3.8.30（`mx-smi` 2.3.1） |
| PyTorch | 2.8.0+metax3.7.0.7 |
| vLLM | 0.24.0+empty |
| Triton | 3.0.0+metax3.7.0.7 |
| FlagGems | **v5.3.5** |
| 预装库 | `mctlassEx`（MetaX CUTLASS 移植，镜像自带） |

### 1.3 基线锚点

| 组件 | 版本 | commit |
|---|---|---|
| vllm-plugin-FL | branch `flagos-2026-s2` | `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0` |
| FlagGems | tag `v5.3.5` | `a7620cc191a0b42e040194622c5758b22a7a25dc` |

---

## 2. 安装

以下命令均在实例上、以 `/workspace` 为工作目录执行。

### 2.1 检查 FlagGems（通常无需重装）

评测镜像已预装 `flag_gems==5.3.5`；本作品**没有修改 FlagGems 算子**，交付包中的 `FlagGems/` 仅用于审计源码一致性：

```bash
python3 -c "import flag_gems; print(flag_gems.__version__)"   # 期望 5.3.5
```

如镜像内版本不符，可自行从源码安装（可选；耗时较长）：

```bash
pip install --no-build-isolation ./FlagGems
```

### 2.2 安装 vllm-plugin-FL（必须）

```bash
pip install --no-deps --no-build-isolation ./vllm-plugin-FL
```

> 使用 `--no-deps` 避免在离线镜像中改动 vLLM/FlagGems 依赖版本；`--no-build-isolation` 保证使用镜像内已编译好的 C++ 扩展工具链。

### 2.3 安装 provenance 自检（建议每次测量前执行）

```bash
# 发行包名为 vllm-plugin-fl（pip 列表里的显示名可能因构建后端而异）
pip show vllm-plugin-fl | grep -i -E "name|location|version"

python3 - <<'PY'
import importlib.metadata as md, json
dist = md.distribution("vllm-plugin-fl")
print("installed at:", dist._path)
print("direct_url:", dist.read_text("direct_url.json"))
PY
# 期望 direct_url 的 url 指向本包 vllm-plugin-FL 的绝对路径
```

若 `direct_url.json` 指向其他仓库，请先 `pip uninstall -y vllm-plugin-fl` 再按 §2.2 安装。

---

## 3. 生成 W8A8 量化 checkpoint（跨平台核心）

### 3.1 为什么 W8A8 是本作品的跨平台核心

本作品只有一项优化在两块卡上都成立：**离线 W8A8 量化 checkpoint**。它把线性层权重从 bf16 压到 int8，产出一份**平台中立**的 packed checkpoint；天数侧用 `torch._int_mm` + 融合 GEMM 承载，沐曦侧用镜像自带的 mctlassEx 承载，见 §7.1 的平台矩阵。其余优化（attention launch 表、prefill TILE、TD 加载、采样 grid）依赖 vLLM 的 Triton attention/采样路径，**只在天数 BI-V150 上生效**；沐曦使用闭源 MACA flash attention，这些补丁不会被安装。因此：

- 若评审侧重“跨平台可迁移性”，W8A8 是主线；
- 天数侧的大幅加速还叠加了平台专用 kernel 调优，这是天数数字高于沐曦侧 W8A8 单项收益的原因；
- 沐曦侧的 W8A8 是当前唯一进入实测的优化项，阶段性 quick 口径已观察到显著正收益（非提交数字，见 §8）。

### 3.2 量化配方与产物

- **权重**：per-channel 对称 RTN；SmoothQuant-lite α=0.5 的通道缩放离线折叠进相邻线性层（RMSNorm / RoPE / SiLU-mul 的等价变换），推理图不变。
- **激活**：不落静态参数，运行时 per-token 动态量化。
- **lm_head**：一并转 int8（−267 MB 权重，decode 侧收益）；embedding 与 norm 保持 bf16。
- **产物格式**：compressed-tensors `pack-quantized`。每个量化 Linear 三键：
  - `weight_packed int32 [N, K/4]`（offset-binary-128，小端低位在前）
  - `weight_scale fp32 [N,1]`
  - `weight_shape int64 [2] = [K, N]`
- **config**：`config.json` 注入 `quantization_config`（`format=pack-quantized`；权重 int8/per-channel/对称/静态；激活 int8/per-token/对称/动态；`targets=["Linear","lm_head"]`）。
- **规模**：42 层 × 7 个投影 + lm_head = **295 个 packed 张量**。
- **单分片约束**：脚本按本赛题 MiniCPM5-2B 的单分片 safetensors 设计；多分片输入会在张量查找处直接报错退出，**不会静默产出错误 ckpt**。输出目录必须为空，避免半成品覆盖。
- 期望产物体积约 2.3–2.6 GB。

### 3.3 生成命令（含强制自检）

转换脚本对固定 seed 幂等，可从原始 bf16 模型一键再生出与提交一致的 int8 checkpoint。**请保留 `--self-check`**（这是保存/加载正确性的第一道门，失败会非零退出）：

```bash
python3 vllm-plugin-FL/tools/quantize_int8_rtn.py \
    --src /workspace/MiniCPM5-2B \
    --dst /workspace/MiniCPM5-2B-w8a8-v2 \
    --smooth-alpha 0.5 \
    --act-stats evidence/act_stats.json \
    --lm-head-int8 --seed 1234 \
    --self-check
```

- `evidence/act_stats.json`：SmoothQuant 折叠因子所用的激活统计。来源：48 条通用文本 prompt、seed=7、共 15,547 个生成 token；采集时 git `9012af6`，**不含 math_500/GSM8K 内容**（与评测集隔离）。provenance 见 `evidence/act_stats_PROVENANCE.json`。
- 期望日志（数值应完全一致/同量级）：

```
RESULT|mode=convert|tensors_packed=295|keep=none|lm_head_int8=True|...
RESULT|check=roundtrip|tensors=295|bit_exact=True
RESULT|check=numeric|sampled=5|worst_scale_rel=5.46e-03|limit=2%
RESULT|check=scheme|layer=PackedW8A8Int8|lm_head=PackedW8A8Int8|cap=71|ok=True
```

> 提示：转换在 CPU 上完成，不需要占用 GPU；建议在正式测量前一次性生成，测量期间不要改写 ckpt。

### 3.4 checkpoint 保存/加载正确性（重点核查）

量化最怕“存进去的是 int8、加载时布局/尺度错位，但程序不报错”。本作品用四道门覆盖保存、加载与端到端：

| 门 | 检查内容 | 证据 |
|---|---|---|
| ① 打包可逆 | 对全部 295 个张量做 `unpack → repack` 逐位比较，要求 `bit_exact=True` | 转换日志 `check=roundtrip`；本地另以 2560×2048、12288×2048、130560×2048、17×4、4×4 等形状复验，含 −128/127 边界，全部逐位一致 |
| ② 数值语义 | 抽 5 层，用 CPU int32 精确累加参考对比“同量化激活 + 反量化权重”的线性输出，按全局幅值归一化 | 实测 `worst_scale_rel=5.46e-3 < 2%`（Metax 实例转化日志，见 `evidence/沐曦_PROVENANCE.txt`） |
| ③ 加载路由 | 走真实 vLLM API：`CompressedTensorsConfig.from_config(...).get_scheme(...)`，确认线性层与 `lm_head` 命中 `CompressedTensorsPackedW8A8Int8`，能力门=71 | 转换日志 `check=scheme`；`packed.py` 的 `install_packed_w8a8_scheme` 与 `int8_mode.should_use_packed_w8a8` |
| ④ 端到端 | 真实 serve + 官方 105 题 | 天数 **0.9714**（基线 0.962）；若布局/尺度错位，精度会直接崩塌。沐曦 quick 运行的输出语义连贯（见 `evidence/沐曦_quick_*`） |

**保存侧（脚本）**：`quantize_int8_rtn.py` 从单分片读取全部张量；对量化站点做折叠、RTN、`pack_int8_offset_binary`，写入三键；未量化张量原样拷贝；`config.json` 注入量化配置；tokenizer / generation_config / index 原样拷贝。

**加载侧（vLLM + 插件）**：

1. 插件导入时由 `vllm_fl/quantization/quant_linear.py` 安装 FL W8A8 kernel 与 packed scheme；
2. vLLM 读取 `quantization_config`，`should_use_packed_w8a8` 按“pack-quantized + 8bit + int + 对称 + per-channel + 动态 per-token”命中；
3. `CompressedTensorsPackedW8A8Int8.create_weights` 注册 `weight_packed`（packed_factor=4、packed_dim=1）与 `weight_scale`；
4. vLLM 的 loader 把分离的 `q/k/v_proj`、`gate/up_proj` packed 权重按输出维切片拼进融合的 `qkv_proj` / `gate_up_proj`；
5. `process_weights_after_loading` 用**精确逆变换**解包（`int32 → uint8 → 减 128 → int8`），注册为 `weight [N,K]`；`weight_shape` 仅作格式完整性元数据，解包使用 `input_size_per_partition` 与 `weight_scale.numel()`，**不依赖其数值**；
6. 天数侧 `FLW8A8DynamicLinearKernel` 显式转置为 `[K,N]` 交给 int8 GEMM；沐曦侧跳过转置，保持 unpack 天然的 `[N,K]` 零拷贝；两侧输出 reshape 都用 `weight_scale.numel()=N` 自适应。

> 结论：保存侧的逐位可逆、加载侧的真实 scheme 路由、以及端到端 105 题精度三者互相独立地验证了 ckpt 没有被“静默写坏或读错”。唯一需要遵守的边界是单分片输入与空输出目录（脚本会强制检查）。

---

## 4. 启动推理服务

> **红线**：serve 命令除 `--model` 指向量化 ckpt 外，与赛题基线命令**逐字一致**；不新增任何 CLI 参数、不设置任何附加环境变量（全部优化已在插件代码内默认开启）。

### 4.1 天数 BI-V150（官方 4.3 命令形态）

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B-w8a8-v2 \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

### 4.2 沐曦 C500（官方 4.3 命令形态，无 `--compilation-config`）

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B-w8a8-v2 \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

### 4.3 启动日志自检

服务就绪后（出现 `Application startup complete`），日志中应能看到以下装载行（主进程与 EngineCore 各一行）：

```
attn_launch_config: 已装载 unified_attention launch 配置表（9 行，...）   # O5
attn_td_mode: 已强制 unified_attention use_td=True（TD 加载路径，...）   # O7
attn_tile_mode: prefill TILE_SIZE 32→16（...）                          # O6
sample_filter_mode: top-p 过滤 grid 已提至 ≤64 programs（...）           # O8
```

W8A8（O1）生效时应出现量化 scheme / kernel 路由日志（天数如
`w8a8→flagos_intmm`，沐曦为 mctlassEx 承载）；也可用 §3.4 的 `check=scheme` 自检直接确认。

### 4.4 快速冒烟测试

```bash
python3 - <<'PY'
import requests, json
r = requests.post('http://127.0.0.1:9031/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={'model': 'minicpm',
          'messages': [{'role': 'user', 'content': 'introduce llm'}],
          'max_tokens': 256})
print(r.json())
PY
```

---

## 5. 官方性能 Benchmark

使用**赛事方提供的 benchmark 脚本原件**（`vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py`，本作品未修改）：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B-w8a8-v2 \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

结果落在 `/workspace/benchmark_results/summary_*.csv`（每场景 4 轮，raw 明细在同目录）。

**判读口径（与赛事基线一致）**：
- 每场景连续 4 轮，**丢弃第 1 轮**（编译/图捕获冷启动），取后 3 轮 `Total tok/s` 的均值。
- 每个场景使用**新鲜 serve** 起测；不要在一个 serve 生命周期内跨场景长跑。
- 关注 `Total tok/s`、`Mean TTFT`；引擎进程若出现静默死亡（`EngineDeadError`），该轮作废并重启服务补测。

### 5.1 天数预期结果（提交口径）

| 场景 | 纯净基线 | **本作品** | 提升 |
|---|---|---|---|
| 4K `[4096,1024,64,256]` | 1879.68 | **3414.58** | **+81.7%** |
| 16K `[16384,1024,64,128]` | 862.06 | **1570.62** | **+82.2%** |

对应 Mean TTFT：4K 12321 ms → **7521 ms**；16K 636 s → **342 s**。

其中 **O1/O2（W8A8）是两平台共用的核心**，单独贡献官方 +6.4% / +5.4%，同时把权重减少 1.84 GB、KV 池扩大约 12%；O5–O8 是叠加在天数专用 kernel/采样路径上的进一步加速。逐轮原始数据见：
- `evidence/天数_4K_final_raw.csv` / `evidence/天数_4K_final_summary.csv`
- `evidence/天数_16K_final_raw.csv` / `evidence/天数_16K_final_summary.csv`

> 结果受 GPU 独占性、镜像编译缓存、机房噪声影响；允许 ±1% 的正常波动。若显著低于上表，请先核对 §2.3 provenance 与 §4.3 日志装载项。

### 5.2 沐曦 C500

沐曦侧适用的是 **O1（W8A8 ckpt）+ O3（mctlassEx 计算后端）**；O2/O5–O8 是天数专用的 Triton kernel/采样优化，不会在沐曦安装。O4（watermark）代码路径虽通用，但 C500 的 64 GB 显存在 16K/64 并发下可整批驻留，不触发容量抢占，实际为 no-op。

适配代码（mctlassEx W8A8 承载、`[N,K]` 零拷贝布局、custom_op 黑盒化、triton<3.3 的 TD stub）已在交付包中并默认按平台自动选择。**官方命令形态（无 `--compilation-config`）下的 4 轮新鲜 serve 数字与 105 题精度正在补测，本版留空**；补测完成后请以同一 `summary_*.csv` 口径填写。阶段性 quick 数据与基线见 `evidence/沐曦_基线_SUMMARY.md`，仅供定位，不作为提交数字。

---

## 6. 精度评测（math_500 Level 3）

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

**预期精度**：`accuracy = 0.9714`（105 题，单轮；基线 0.962，红线 ≥0.95）。判分原件见 `evidence/天数_math500_L3_report.json`。

> `temperature=1.0` 采样存在轮间方差（基线 0.962 同为单轮口径）。复现结果在 0.95 以上即满足赛题要求；若低于 0.95，请先确认是否误关了 O6/O7（见 §7 回退开关）或使用的不是交付版插件。

---

## 7. 优化清单、平台覆盖与回退开关

### 7.1 平台覆盖矩阵（先看这张表）

| 编号 | 优化 | 天数 BI-V150 | 沐曦 C500 | 说明 |
|---|---|---|---|---|
| **O1** | **离线 W8A8 量化 ckpt** | ✅ | ✅ | **两平台共用同一份平台中立 packed ckpt**；权重 −1.84 GB |
| O2 | 天数 W8A8 计算后端（`torch._int_mm` + 融合 GEMM/epilogue） | ✅ | — | 天数的 int8 GEMM 载体 |
| O3 | 沐曦 W8A8 计算后端（mctlassEx） | — | ✅ | 沐曦的 int8 GEMM 载体；与 O1 组合 |
| O4 | 调度准入 watermark=0.06 | ✅ 实测有效 | ⚪ 代码路径生效 | 沐曦 64 GB、16K 64 并发可整批驻留，无容量钉死，实际 no-op |
| O5 | attention launch 配置表 | ✅ | — | 补丁目标是 vLLM Triton `unified_attention`；沐曦用 MACA flash |
| O6 | prefill TILE=16 | ✅ | — | 同上 |
| O7 | TD tensor-descriptor 加载 | ✅ | — | 沐曦 triton 3.0 无 `make_tensor_descriptor`，TD stub 强制 False |
| O8 | 采样过滤 grid 16→64 | ✅ | — | 沐曦的 Triton topk/topp 编译失败，走 PyTorch fallback |
| R1 | 自研 split-KV kernel | 研究档案（**天数侧未接线**） | hop 默认关 | 天数侧 env 不生效，防误开；沐曦 hop 独立，`VLLM_FL_ATTN_SPLITKV=1` 才启用 |

**一句话总结**：O1 是唯一跨平台的核心；O2/O3 是同一 W8A8 思路在两块卡上的不同 GEMM 载体；O5–O8 是天数专有的 kernel/采样调优；O4 是框架级调度项，只在容量受限的天数 16K 上产生实测收益。


### 7.2 清单与回退开关

全部采纳项均为**代码级默认开启**（官方命令无需任何 env）。每个改动都保留显式回退开关，便于评审消融：

| 编号 | 优化 | 默认 | 回退开关 | 数值性质 |
|---|---|---|---|---|
| **O1** | W8A8 量化 ckpt（SmoothQuant-lite α=0.5 + per-channel RTN + lm_head int8） | 由 `--model` 指向 ckpt 激活 | 改回原 bf16 模型路径 | 精度门已过（0.9714） |
| O2 | 天数 W8A8：裸启动量化 + `torch._int_mm` + 融合 epilogue + 融合 decode GEMM | 开 | `VLLM_FL_INTMM_FUSED_GEMM=0` / `VLLM_FL_INTMM_FUSED_DECODE=0` | 位精确（对齐 int32 参考） |
| O3 | 沐曦 mctlassEx W8A8 承载（按平台自动） | 自动 | `VLLM_FL_MCTLASS_W8A8=0` | int8 数值 |
| O4 | 调度准入 watermark=0.06（16K 零抢占） | 开 | `VLLM_FL_WATERMARK=0` | 调度行为，精度等价 |
| O5 | attention launch 参数表（路径×bs档×kv档） | 开 | `VLLM_FL_ATTN_LAUNCH=0` | **bit-exact** |
| O6 | prefill attention TILE 32→16 | 开 | `VLLM_FL_ATTN_TILE=0` | 非 bit-exact（softmax 等价） |
| O7 | Q/K/V tensor-descriptor 加载（use_td=True） | 开 | `VLLM_FL_ATTN_TD=0` | 非 bit-exact（softmax 等价） |
| O8 | top-p 采样过滤 grid 16→64（自管缓冲） | 开 | `VLLM_FL_SAMPLE_FILTER=0` | **bitwise 等价** |
| R1 | 自研 decode split-KV kernel（研究档案） | **不可达** | 天数侧未接线；沐曦 hop 用 `VLLM_FL_ATTN_SPLITKV=1` | 非 bit-exact |
| — | KV cache int8 扩池（16K 判负，未启用） | **关** | `VLLM_FL_KV_INT8=1` 开启 | 1/16 分歧 |

> O6/O7 非 bit-exact 是**预期内**的数值路径变化（加载/累加顺序不同，softmax 数学等价）；二者已通过 105 题精度门与 16 题文本对拍记录。O5 与 O8 为逐位等价。

---

## 8. 沐曦 C500 数据状态（本版留空）

交付代码已包含沐曦适配（`vllm_fl/dispatch/backends/vendor/metax/`、`quantization/w8a8/mctlass_mm.py`、`metax/patches/`），并按官方命令形态（**无** `--compilation-config`）具备可运行性。但以下两项数据在打包时**仍在测试中**，本版按要求留空：

| 项 | 状态 |
|---|---|
| 沐曦 C500 4K 场景 Total tok/s（官方 4 轮丢首） | 测试中，待填 |
| 沐曦 C500 16K 场景 Total tok/s（官方 4 轮丢首） | 测试中，待填 |
| 沐曦 C500 math_500 L3 accuracy（≥0.95） | 测试中，待填 |
| 沐曦基线（团队内部口径：bf16 + 基线插件 13eb9be） | 4K 3786.30 / 16K 5583.49（quick 口径，见 evidence） |

评测时如先跑沐曦，请沿用与 §5 相同的脚本与判读口径；`mctlassEx` 不可用的镜像会自动回退到 `torch._int_mm` / FlagGems `scaled_mm` 链（日志中可见回退原因）。

---

## 9. 常见问题与注意事项

1. **EngineCore 间歇性静默退出**：在长 bench 中偶见（无 Python 栈）。对策：每场景独立新鲜 serve、4 轮内死亡则作废该轮并重启补测，不要把两场景串在同一 serve 内。
2. **`pip install` 装错来源**：务必用 `direct_url.json` 核对；从旧仓库或其他路径安装会带入非交付改动，导致数字不可复现。
3. **量化 ckpt 缺失/被改写**：重新执行 §3；脚本幂等，固定 seed 下逐位再生。
4. **首次推理慢**：图捕获与 Triton 编译发生在启动/首轮，官方口径丢弃首轮即为此设计；请勿用首轮数字做比较。
5. **精度评测超时**：16K 长上下文 + 32K max_tokens 的样本耗时较长，保持 `--timeout 3600 --ignore-errors`，并确保 serve 在整个评测期间存活。
6. **不要在测量期间设置回退 env**：所有回退开关仅用于消融；正式评测应保持默认（官方命令零附加 env）。

---

## 10. 合规性声明

- 仅使用 MiniCPM5-2B，无其他模型、无投机采样、无 serve 侧量化超参。
- 官方 serve 参数逐字未改，`--model` 仅指向由原模型幂等再生的自研量化 checkpoint（赛题允许的“自行开发的量化策略”）。
- 官方 benchmark / evalscope 脚本原件未修改；无 benchmark 脚本调优。
- 未删除或绕过框架的算子选择逻辑：新增 W8A8 dispatch op 采用“mctlassEx（沐曦）/ `torch._int_mm`（天数）优先 + FlagGems `scaled_mm` 有序兜底”的显式优先级，并在不可用时自动回退。
- 所有优化均可在提交代码中定位（见技术报告附录 A 的文件清单），未合并上游最新分支代码。
- FlagGems 保持 v5.3.5 原样，未做算子替换式“优化”。

---

## 11. 结果校验速查

| 校验项 | 期望 |
|---|---|
| `flag_gems.__version__` | `5.3.5` |
| `pip show vllm-plugin-fl` 的 location + `direct_url.json` | 指向本包 `vllm-plugin-FL` |
| 转换日志 `check=roundtrip` | `tensors=295 | bit_exact=True` |
| 转换日志 `check=scheme` | 线性层与 lm_head 均 `PackedW8A8Int8`，`cap=71` |
| serve 日志装载行 | O5/O6/O7/O8 均出现；W8A8 路由已生效 |
| 4K `Total tok/s`（天数，丢首均值） | ≥ 3380（3414.58 ±1%） |
| 16K `Total tok/s`（天数，丢首均值） | ≥ 1550（1570.62 ±1%） |
| math_500 L3 accuracy | ≥ 0.95（期望 0.9714） |
| 沐曦三项 | 待补（见 §8） |
