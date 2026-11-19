# 证据文件索引

本目录保存提交用 benchmark / 精度原始结果及关键 provenance。所有性能数字均来自赛事方
`benchmark_throughput_serve.py` 原件；精度数字来自赛事指定 evalscope math_500 Level 3。

## 天数 BI-V150（提交口径）

| 文件 | 说明 |
|---|---|
| `天数_4K_final_summary.csv` | 终选配置（W8A8 ckpt + ②+⑤+①，代码默认）4K 场景 summary：**Total tok/s = 3414.58**，Mean TTFT = 7520.8 ms |
| `天数_4K_final_raw.csv` | 同场景 4 轮 raw（首轮丢弃；后 3 轮 3010.98 / 3576.64 / 3656.12） |
| `天数_16K_final_summary.csv` | 终选配置 16K 场景 summary：**Total tok/s = 1570.62**，Mean TTFT = 342358.36 ms |
| `天数_16K_final_raw.csv` | 同场景 4 轮 raw（首轮丢弃；后 3 轮 1572.97 / 1569.31 / 1569.58） |
| `天数_math500_L3_report.json` | evalscope 105 题判分原件，**accuracy = 0.9714**（红线 0.95，基线 0.962） |

## 阶段消融（用于理解各项优化量级，不作为提交数字）

| 文件 | 说明 |
|---|---|
| `stages/O5_summary.csv` | +O5 attention launch 表：4K 2658.72 / 16K 1262.66 |
| `stages/O4_summary.csv` | +O4 watermark：4K 2668.42 / 16K 1276.62 |
| `stages/W8A8_run1_summary.csv` / `W8A8_run2_summary.csv` | +O1/O2 W8A8：4K ≈2838 / 16K ≈1346（两次独立官方 run） |
| `stages/O8_summary.csv` | 仅 O8（采样 grid）：4K 2741.07（存在未归因轮间波动）/ 16K 1481.00 |

## 量化与复现依赖

| 文件 | 说明 |
|---|---|
| `act_stats.json` | SmoothQuant 折叠激活统计（48 prompts / seed 7 / 15,547 tokens；不含 math_500/GSM8K 内容） |
| `act_stats_PROVENANCE.json` | 采集时间、git、prompt 数、seed |
| `沐曦_PROVENANCE.txt` | 含转换自检摘要：295 个张量 round-trip bit-exact；数值 `worst_scale_rel=5.46e-3 < 2%`；scheme=`PackedW8A8Int8`（含 lm_head） |

转换命令见交付包根目录 `readme.md` §3（务必保留 `--self-check`）；脚本固定 seed，可幂等再生同一 ckpt。
保存/加载的四道验证门（打包可逆、数值语义、加载路由、端到端精度）见 `readme.md` §3.4。

## 沐曦 C500（测试中，本版留空）

| 文件 | 说明 |
|---|---|
| `沐曦_基线_SUMMARY.md` | 团队内部 quick 口径的基线与 w8a8-mctlass 过程数据。**注意**：该批数据使用了带 `FULL_DECODE_ONLY` 的非官方命令形态，且轮次不完整，仅用于定位，不作为提交数字 |
| `沐曦_quick_*.csv` | 对应的 raw / summary CSV |
| `沐曦_PROVENANCE.txt` | 沐曦实例的软硬件版本、插件安装路径与 ckpt 生成记录 |

官方命令形态（无 `--compilation-config`）下的 4K/16K 四轮丢首均值与 105 题精度正在补测，
完成后应替换为正式口径文件，并在报告 §5.5 与 readme §8 回填。
