# tomato_vote8_outputs

这个目录保存了一次 `vote=8` 的 ToMATO/Tomato 推理导出结果。当前这批结果对应：

- 处理样本数：`500`
- worker 数：`10`
- 每题并行 solver 次数：`8`
- judge 模型：`qwen/qwen3-32b`

上面的统计来自同目录下的 `run_summary.json`。

## 这两个文件的关系

本 README 主要说明下面两个文件：

- `tomato_vote8_raw.jsonl`
- `tomato_vote8_all_wrong.jsonl`


| 文件 | 行数 | 含义 |
| --- | ---: | --- |
| `tomato_vote8_raw.jsonl` | `448` | 主结果集。每条样本至少有 `1` 个 solver attempt 被 judge 判为正确。 |
| `tomato_vote8_all_wrong.jsonl` | `52` | 失败集。每条样本都没有任何 parseable solver attempt 判对。 |


`all_wrong` 不是 `raw` 的子集，而是和 `raw` 互补。

## 1. tomato_vote8_raw.jsonl

### 适合做什么

- 看主结果和保留下来的可用样本
- 分析 8 次投票里“对了几次”
- 抽取 winning answer、SIP 推理、natural reasoning
- 后续构造参考集、SFT 数据或做误差分层

### 一条记录大致包含什么

`raw` 中常见的顶层字段有：

- 基础题目信息：`id`, `story_id`, `dimension`, `order`, `task_type`, `full_story`, `story_summary`, `question`
- 标准答案：`correct_answer_text`, `correct_answer_aliases`
- 结构化标注：`state`, `action`, `meta`
- 8 路 solver 统计：`num_parallel_solver_attempts`, `num_parseable_solver_attempts`, `num_correct_solver_attempts`, `num_wrong_solver_attempts`
- attempt 分桶：`solver_attempts`, `correct_solver_attempts`, `wrong_solver_attempts`, `solver_failures`
- winning 结果：`winning_solver_attempt_id`, `winning_solver_answer`, `winning_solver_reasoning`, `winning_solver_thinking`, `winning_solver_raw_model_text`, `winning_solver_judge_result`
- SIP / natural 推理：`silver_sip_reasoning`, `natural_reasoning_paragraph`, `rewrite_source_trace`, `raw_model_text_sip`
- 运行与模型信息：`model_name_solver`, `model_name_judge`, `model_name_sip`, `model_name_natural`, `usage_solver_winner`, `usage_judge_winner`, `usage_sip`, `usage_natural`, `worker_id`

其中：

- `winning_solver_answer` 在当前导出里只有一个字段：`final_answer`
- `silver_sip_reasoning` 下面包含 `cue_encoding`, `cue_interpretation`, `goal_clarification`, `response_generation`, `natural_cot`

### 这个文件里的样本质量分布

`raw` 的每条样本都至少有一次判对，因此：

- `num_correct_solver_attempts` 取值范围是 `1` 到 `8`
- 当前分布是：

| num_correct_solver_attempts | 样本数 |
| ---: | ---: |
| 1 | 13 |
| 2 | 10 |
| 3 | 8 |
| 4 | 9 |
| 5 | 8 |
| 6 | 20 |
| 7 | 41 |
| 8 | 339 |

因此 `raw` 更像“保留下来的主样本集”，而不是所有 500 条样本的原始无筛选拼接。

## 2. tomato_vote8_all_wrong.jsonl

### 适合做什么

- 专门分析失败案例
- 看 8 次并行尝试为什么全错
- 排查 prompt、judge、解析或任务本身的难点
- 回收 hard case 做再训练或人工复审

### 一条记录大致包含什么

`all_wrong` 的顶层字段相对精简，主要包括：

- 基础题目信息：`id`, `story_id`, `dimension`, `order`, `task_type`, `full_story`, `question`
- 标准答案：`correct_answer_text`, `correct_answer_aliases`
- 错误标记与统计：`error`, `num_parallel_solver_attempts`, `num_parseable_solver_attempts`, `num_correct_solver_attempts`, `num_wrong_solver_attempts`
- attempt 轨迹：`solver_attempts`, `correct_solver_attempts`, `wrong_solver_attempts`, `solver_failures`
- 运行信息：`worker_id`

当前这 `52` 条记录的 `error` 都是同一个值：

```text
no_correct_attempt_among_parseable_solver_calls
```

这说明这些样本并不是“没跑起来”，而是“能解析、能判分，但没有任何一个 parseable answer 被判为正确”。

### solver_attempts 里有什么

`all_wrong` 里每个 `solver_attempts[i]` 一般包含：

- `attempt_id`
- `final_answer`
- `free_form_reasoning`
- `thinking`
- `raw_model_text`
- `parsed_by_fallback`
- `judge_result`
- `judge_raw_text`
- `is_correct`
- `usage`
- `usage_judge`

这部分最适合拿来做错误归因，因为每个 attempt 的原始输出和 judge 判断都还在。

## 两个文件的核心区别

| 维度 | `raw` | `all_wrong` |
| --- | --- | --- |
| 是否保留样本 | 是，主结果集 | 是，失败集 |
| 样本条件 | 至少 1 次判对 | 0 次判对 |
| 是否有 `error` 字段 | 通常没有 | 有 |
| 是否含 `state` / `action` / `meta` | 有 | 没有 |
| 是否含 winning solver 字段 | 有 | 没有 |
| 是否含 SIP / natural reasoning | 有 | 没有 |
| 主要用途 | 主分析、构造数据 | 误差分析、回收 hard case |

