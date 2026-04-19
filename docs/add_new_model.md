# 新增模型指南

本指南介绍如何在 TomTest 框架中使用新模型进行评测。

## 概述

TomTest 支持任何兼容 OpenAI API 的模型。添加新模型只需：

1. 修改 `experiment_config.yaml` 配置
2. 运行评测脚本

## 配置文件

所有实验参数统一在 `experiment_config.yaml` 中配置：

```yaml
llm:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://localhost:8000/v1
  temperature: 0.6
  max_tokens: 32768
  max_workers: 64
  enable_thinking: false
  system_prompt: ""

judge:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://localhost:8000/v1
  temperature: 0.0
  max_tokens: 4096
  enable_thinking: false
  use_llm_judge: false

repeats: 1
max_samples: 100
datasets_path: datasets
results_path: results
```

## 模式 1：使用 vLLM（本地模型）

### 1.1 启动 vLLM serve

```bash
vllm serve /path/to/your/model \n    --port 8000 \n    --tensor-parallel-size 1
```

### 1.2 配置实验参数

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: my-model
  api_key: not-needed
  api_url: http://localhost:8000/v1
  temperature: 0.6
  max_tokens: 32768
  max_workers: 64
  enable_thinking: false
```

### 1.3 运行评测

```bash
# 评测单个数据集
python tasks/ToMChallenges/run.py

# 评测所有数据集
python run_all.py
```

## 模式 2：使用云端 API

### 2.1 DeepSeek

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
```

运行：

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_all.py
```

### 2.2 OpenAI

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_url: https://api.openai.com/v1
```

运行：

```bash
export OPENAI_API_KEY="sk-xxx"
python run_all.py
```

### 2.3 其他兼容 OpenAI API 的服务

| 服务 | API URL | Model 名称示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-coder` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |

## 配置说明

### LLM 配置

| 字段 | 说明 | 默认值 |
|---|---|---|
| `llm.model_name` | 模型名称 | - |
| `llm.api_url` | API 端点 URL | - |
| `llm.api_key` | API 密钥（支持环境变量 `${VAR_NAME}`） | - |
| `llm.temperature` | 采样温度（0.0 = 确定性，1.0 = 随机） | `0.6` |
| `llm.max_tokens` | 最大输出 token 数 | `32768` |
| `llm.max_workers` | 最大线程数 | `64` |
| `llm.enable_thinking` | 是否启用思考模式 | `false` |
| `llm.system_prompt` | 系统 prompt | `""` |

### Judge 配置（可选）

| 字段 | 说明 | 默认值 |
|---|---|---|
| `judge.model_name` | Judge 模型名称 | - |
| `judge.api_url` | Judge API URL | - |
| `judge.api_key` | Judge API 密钥 | - |
| `judge.temperature` | Judge 温度（通常为 0.0） | `0.0` |
| `judge.max_tokens` | Judge 输出 token 数 | `4096` |
| `judge.use_llm_judge` | 是否启用 LLM judge | `false` |

### 实验参数

| 字段 | 说明 | 默认值 |
|---|---|---|
| `repeats` | 重复运行次数（取平均） | `1` |
| `max_samples` | 最大样本数（0 = 全部，>0 = 随机抽样） | `0` |
| `datasets_path` | 数据集根目录 | `datasets` |
| `results_path` | 结果输出目录 | `results` |

## 结果生成与查看

### 生成表格

```bash
# 从 results 生成各数据集表格
# 不指定 --exp-suffix 时，自动选择每个数据集最新的实验
python generate_dataset_tables.py

# 指定实验后缀
python generate_dataset_tables.py --exp-suffix 20240101_120000

# 生成总览汇总
python generate_summary.py
```

### 查看结果

所有结果保存在 `results/` 目录：

```bash
# 列出所有结果
ls -lh results/

# 查看最新结果
cat results/$(ls -t results/*/*/exp_*/metrics.json | head -1) | jq

# 比较不同模型的结果
for f in results/*/*/exp_*/metrics.json; do
    echo "$f: $(cat $f | jq -r '.avg_metrics.accuracy')"
done
```

## 调试技巧

### 1. 使用少量样本快速验证

编辑 `experiment_config.yaml`：

```yaml
max_samples: 3  # 只测试 3 条样本（随机抽取）
```

### 2. 检查 API 连接

```bash
curl http://localhost:8000/v1/models
```

## 常见问题

### Q: 提示 "Connection refused"

确保 vLLM 服务正在运行：

```bash
curl http://localhost:8000/v1/models
```

### Q: API 密钥错误

检查环境变量是否正确设置：

```bash
echo $DEEPSEEK_API_KEY
```

### Q: 模型不支持结构化输出

框架会自动检测模型是否支持结构化输出。如果不支持，会自动降级到 JSON object 模式。

### Q: 如何设置不同的温度？

编辑 `experiment_config.yaml`：

```yaml
llm:
  temperature: 0.1  # 更确定的输出
```

或

```yaml
llm:
  temperature: 0.9  # 更随机的输出
```

### Q: 随机抽样是否可复现？

框架内部不使用全局随机种子，每个数据集的 `load_and_limit_data` 函数使用固定种子保证可复现。

### Q: 如何使用 LLM Judge？

编辑 `experiment_config.yaml`：

```yaml
judge:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  use_llm_judge: true  # 启用 LLM Judge
```

然后在数据集的 `run.py` 中调用 `runner.create_judge_client()` 获取 judge 客户端。