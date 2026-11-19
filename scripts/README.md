# 辅助工具与探针

这些脚本用于复核优化机理、再生配置表或做消融，**不参与官方评测流程**。官方性能/精度评测
只需按根目录 `readme.md` 的步骤执行。

| 文件 | 用途 | 依赖 |
|---|---|---|
| `sweep_attn_launch.py` | 扫描 attention 的 launch 配置（num_warps/num_stages），以 bit-exact 为门禁再生 O5 配置表；输出 `sweep_table.py` / `sweep_table.json` | GPU + 已安装插件 |
| `micro_bench_attn_launch.py` | O5 的代表形状 micro-bench（默认臂 vs 调优臂，bit-exact 自检） | GPU + 已安装插件 |
| `quant/common.py` | **量化转换脚本的必需依赖**：SmoothQuant 折叠数学、模型结构常量、keep 预设 | 仅 Python 标准库 |
| `probe/parity_check.py` | 对 serve 发 16 条 greedy 请求，与基线 JSON 逐条对比（文本对拍） | 运行中的 serve |
| `probe/sample_filter_bench.py` | O8 采样过滤两条实现的同口径计时与逐位等价性检查 | GPU + 已安装插件 |

说明：

- `sweep_attn_launch.py` 会按 `--out` 写出新表与 JSON；用 `VLLM_FL_ATTN_LAUNCH_TABLE=<json>`
  可在不改代码的前提下做 A/B。变更 Triton/flagtree 版本后应重跑并替换 `LAUNCH_TABLE`。
- `probe/sample_filter_bench.py` 已改为相对定位插件源码，无需再改路径。
- `quant/common.py` 同时被 `vllm-plugin-FL/tools/quantize_int8_rtn.py` 通过包根目录导入，
  请勿移动或重命名。
