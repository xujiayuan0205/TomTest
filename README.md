# TomTest

基于结构化输出的 Theory-of-Mind（心智理论）评测框架，支持多数据集、多模型的基准评测。

## 设计理念

**结构化输出优先** - 使用 Pydantic 定义输出 Schema，直接从结构化对象获取答案，避免复杂的字符串解析：
- **新增模型时**：只需修改 `experiment_config.yaml` 中的 API 配置
- **新增数据集时**：复用现有 Schema，只需编写 `prompts.py` 和 `metrics.py`
- **无需字符串处理**：结构化输出自动解析，代码简洁可靠

## 目录结构

```
TomTest/
├── datasets/                    # 数据集（Arrow 格式）
│   ├── ToMChallenges/
│   ├── SocialIQA/
│   └── ...
├── tasks/                       # 数据集评测代码
│   ├── ToMChallenges/
│   │   ├── config.yaml          # 数据集配置
│   │   ├── prompts.py          # Prompt 模板
│   │   ├── metrics.py          # 指标计算
│   │   └── run.py             # 评测脚本
│   ├── SocialIQA/
│   └── ...
├── src/                         # 核心框架代码
│   ├── llm/
│   │   ├── __init__.py         # 导出 LLMClient, StructureClient
│   │   ├── client.py           # LLMClient 基类
│   │   └── structure_client.py  # 结构化输出客户端
│   ├── dataloader/
│   │   └── dataloader.py      # 数据集加载器
│   ├── schemas.py             # 统一的 Schema 定义
│   ├── runner.py              # 评测运行器公共函数
│   └── utils.py              # 通用工具函数
├── results/                     # 评测结果输出
│   └── {dataset}/{model}/exp_{timestamp}/
│       ├── config.json         # 完整配置
│       ├── metrics.json        # 指标（avg + all）
│       └── prediction.jsonl    # 详细预测结果
├── tables/                      # 评测结果表格
│   ├── SUMMARY.md              # 总览表格
│   └── {dataset}/
│       ├── 基础指标.md
│       └── 其他指标.md
├── experiment_config.yaml        # 实验配置（全局）
├── run_all.py                  # 统一运行入口
├── generate_dataset_tables.py   # 生成数据集表格
└── generate_summary.py          # 生成总览汇总
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai pyyaml tqdm datasets pyarrow
```

### 2. 配置实验参数

编辑 `experiment_config.yaml`：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}  # 支持环境变量
  api_url: https://api.deepseek.com/v1
  temperature: 0.6
  max_tokens: 32768
  max_workers: 64
  enable_thinking: false
  system_prompt: ""

judge:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  temperature: 0.0
  max_tokens: 4096
  use_llm_judge: false

repeats: 3
max_samples: 0  # 0 = 全部样本，>0 = 随机抽样
datasets_path: datasets
results_path: results
```

### 3. 运行评测

```bash
# 运行所有数据集
python run_all.py

# 或单独运行某个数据集
python tasks/ToMChallenges/run.py
```

### 4. 生成结果表格

```bash
# 从 results 生成各数据集表格（--exp-suffix 可选）
# 不指定 --exp-suffix 时，自动选择每个数据集最新的实验
python generate_dataset_tables.py

# 指定实验后缀
python generate_dataset_tables.py --exp-suffix 20240101_120000

# 生成总览汇总
python generate_summary.py

# 或直接输出到终端
python generate_summary.py --stdout
```

## 支持的数据集

| 数据集 | Schema | 说明 |
|---|---|---|
| ToMChallenges | `MCQAnswer` | A/B 二选一 ToM 基准测试 |
| SocialIQA | `MCQAnswer3` | 社交情境理解（三选一） |
| Belief_R | `MCQAnswer3Lower` | 信念追踪（小写 a/b/c） |
| RecToM | `MultiLabelAnswer` | 推荐中的心智理论（多选） |
| ToMQA | `OpenAnswer` | 开放式问答 |
| Tomato | `MCQAnswer` | 多选题（支持选项 shuffle） |
| ToMBench | `MCQAnswer` | ToM 基准测试（中英文） |

## 可用的 Schema

所有 Schema 统一在 `src/schemas.py` 中定义：

| Schema | 使用场景 | 字段 |
|---|---|---|
| `MCQAnswer` | A/B/C/D 四选一 | `answer: Literal["A", "B", "C", "D"]` |
| `MCQAnswer3` | A/B/C 三选一 | `answer: Literal["A", "B", "C"]` |
| `MCQAnswer3Lower` | a/b/c 三选一小写 | `answer: Literal["a", "b", "c"]` |
| `OpenAnswer` | 开放式问答 | `answer: str` |
| `OneWordAnswer` | 单词回答 | `answer: str`（无空白字符） |
| `JudgeAnswer` | LLM Judge 判断 | `answer: Literal["True", "False"]` |
| `MultiLabelAnswer` | 多标签多选 | `answer: List[str]` |

## 配置文件说明

### 数据集配置（`tasks/{dataset}/config.yaml`）

```yaml
dataset: ToMChallenges
path: ToMChallenges/test
method: ZS_vanilla
schema: MCQAnswer
system_prompt: ""  # 可选，覆盖 experiment_config.yaml 中的设置
```

### 实验配置（`experiment_config.yaml`）

| 参数 | 说明 | 默认值 |
|---|---|---|
| `llm.model_name` | 模型名称 | - |
| `llm.api_url` | API 地址 | - |
| `llm.api_key` | API 密钥（支持环境变量） | - |
| `llm.temperature` | 采样温度 | `0.6` |
| `llm.max_tokens` | 最大输出 token 数 | `32768` |
| `llm.max_workers` | 最大线程数 | `64` |
| `llm.enable_thinking` | 是否启用思考模式 | `false` |
| `llm.system_prompt` | 系统 prompt | `""` |
| `judge.use_llm_judge` | 是否使用 LLM judge | `false` |
| `repeats` | 重复运行次数 | `1` |
| `max_samples` | 最大样本数（0=全部） | `0` |
| `datasets_path` | 数据集根目录 | `datasets` |
| `results_path` | 结果输出目录 | `results` |

## 公共函数（`src/runner.py`）

| 函数 | 说明 |
|---|---|
| `load_dataset_config()` | 加载数据集配置 |
| `load_experiment_config()` | 加载实验配置 |
| `load_schema()` | 根据名称加载 Schema 类 |
| `create_llm_client()` | 创建 LLM 客户端 |
| `create_judge_client()` | 创建 Judge 客户端 |
| `load_and_limit_data()` | 加载数据并支持随机抽样 |
| `save_common_results()` | 保存评测结果 |
| `print_summary_stats()` | 打印统计摘要 |

## 结果文件结构

**results/ 结构**：
```
results/
├── {dataset}/
│   └── {model}/
│       └── exp_{timestamp}/
│           ├── config.json         # 所有配置
│           ├── metrics.json        # {avg_metrics, all_metrics}
│           └── prediction.jsonl    # 每行一个样本的预测结果
```

**tables/ 结构**：
```
tables/
├── SUMMARY.md              # 总览表格（所有数据集 × 模型 × accuracy）
└── {dataset}/
    ├── 基础指标.md          # accuracy、correct、total
    └── 其他指标.md           # 其他所有指标
```

## 扩展指南

- [新增数据集指南](docs/add_new_dataset.md) - 如何添加新的评测数据集
- [新增模型指南](docs/add_new_model.md) - 如何使用新模型进行评测

## 许可证

MIT License
