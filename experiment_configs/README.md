# 多模型实验配置（方案 A）

与仓库根目录 `experiment_config.yaml` 格式相同；每个文件对应一个**本地 vLLM** 的 `model_name` 与独立 **`results_path`**，避免多轮评测互相覆盖。

## 当前包含的 5 个模型（不含 DeepSeek）

与 `/home/xujy/TomTest_copy/scripts/run_five_models_*.sh` 一致：

| 配置文件 | served-model-name |
|----------|-------------------|
| `Qwen3-0.6B.yaml` | Qwen3-0.6B |
| `Qwen3-4B.yaml` | Qwen3-4B |
| `Qwen3-8B.yaml` | Qwen3-8B |
| `gemma-3-4b-it.yaml` | gemma-3-4b-it |
| `Meta-Llama-3.1-8B-Instruct.yaml` | Meta-Llama-3.1-8B-Instruct |

默认 `api_url` 为 `http://127.0.0.1:8006/v1`，需与 vLLM `--port` 一致。

## 用法

### 自动化（起 vLLM + 跑 ToMBench + Tomato）

在仓库根目录：

```bash
bash scripts/run_five_local_models_tombench_tomato.sh
```

环境变量（可选）：

- `GPU_ID`：默认 `6`
- `PORT`：默认 `8006`（须与各 yaml 中 `api_url` 端口一致，或自行改 yaml）
- `QWEN06`、`QWEN4B`、`QWEN8B`、`GEMMA`、`LLAMA`：权重目录，与 TomTest_copy 默认相同
- `VLLM_EXTRA`：附加传给 `vllm serve` 的参数
- `RUN_VLLM=0`：不启动 vLLM，仅 `cp` 配置并运行评测（需已自行在对应端口起好服务）
- `VLLM_LOG=/path/to.log`：可选，把各轮 `vllm serve` 输出追加到该文件；不设则丢弃 vLLM 进程日志，**ToMBench/Tomato 输出仍打印到终端**（可用 `nohup ... > out.log 2>&1` 统一保存）

### 手动单次

```bash
cp experiment_configs/Qwen3-8B.yaml experiment_config.yaml
python ToMBench/run.py
python Tomato/run.py
```

结果目录见各 yaml 中 `results_path`（如 `results/Qwen3-8B`）。
