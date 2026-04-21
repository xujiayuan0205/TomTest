# SocialIQA vote8 输出说明

这个目录里最核心、最值得一起提供的两份文件是：

- `socialiqa_vote8_raw.jsonl`
- `socialiqa_vote8_all_wrong.jsonl`

它们来自同一条 `vote8 + judge + SIP` 产线，但保存的是两类互补样本：

- `raw`：至少有一个并发 solver 尝试被判为正确，因此可以继续生成 SIP 重写。
- `all_wrong`：有可解析的 solver 结果，但所有可解析结果都被判为错误，因此没有进入 SIP 重写。

按当前目录下这次运行的 `run_summary.json`：

- 输入数据：`socialiqa_550.json`
- 处理区间：`start_index=0`, `end_index=500`
- 处理样本数：`500`
- 并行 solver 次数：`8`
- `socialiqa_vote8_raw.jsonl`：`232` 条
- `socialiqa_vote8_all_wrong.jsonl`：`268` 条
- `socialiqa_vote8_errors.jsonl`：`0` 条

- 两个文件合起来覆盖了全部处理样本
- 没有样本因为“完全无法解析 solver 输出”而进入 `errors`

## 1. 整体流程

对应脚本：

- `socialiqa_vote8_parallel_launcher.py`
- `socialiqa_vote8_parallel_worker.py`

单条样本的大致处理逻辑是：

1. 对每条 SocialIQA 样本并发跑 `8` 次 solver。
2. 从每次 solver 输出中解析 `final_answer` 和简短 `reasoning`。
3. 用 judge 模型把 solver 的 `final_answer` 与 `correct_answer_text` / `correct_answer_aliases` 做语义等价判断。
4. 只要存在至少一个正确 solver 轨迹，这条样本就写入 `socialiqa_vote8_raw.jsonl`。
5. 选中一个正确轨迹作为 `winner`，再生成结构化 SIP 推理和自然语言推理段落。
6. 如果所有可解析 solver 轨迹都错误，这条样本写入 `socialiqa_vote8_all_wrong.jsonl`。
7. 如果一个可解析结果都没有，样本才会进入 `socialiqa_vote8_errors.jsonl`。

## 2. 两个文件的分工

| 文件 | 收录条件 | 是否包含 SIP 重写 | 主要用途 |
| --- | --- | --- | --- |
| `socialiqa_vote8_raw.jsonl` | 至少一个 solver 尝试判对 | 是 | 保留成功样本、winning trace、SIP 骨架、自然推理段落 |
| `socialiqa_vote8_all_wrong.jsonl` | 所有可解析 solver 尝试都判错 | 否 | 做失败分析、看 vote8 没覆盖到哪些题 |

最容易误解的一点是：

- `raw` 不是“所有原始采样结果”，而是“成功进入 SIP 重写阶段的样本全集”。
- `all_wrong` 不是异常文件，而是和 `raw` 互补的失败子集。

## 3. `socialiqa_vote8_raw.jsonl` 结构

`raw` 的每一行对应一条至少命中过一次正确答案的样本，顶层字段大致可以分成几组。

### 3.1 原始样本信息

- `id`
- `story_id`
- `dimension`
- `order`
- `task_type`
- `full_story`
- `story_summary`
- `question`
- `correct_answer_text`
- `correct_answer_aliases`
- `wrong_answer_texts`
- `state`
- `action`
- `meta`

### 3.2 vote8 求解统计

- `num_parallel_solver_attempts`
- `num_parseable_solver_attempts`
- `num_correct_solver_attempts`
- `num_wrong_solver_attempts`
- `solver_attempts`
- `correct_solver_attempts`
- `wrong_solver_attempts`
- `solver_failures`

其中 `solver_attempts` 里的常见字段包括：

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

### 3.3 winning solver 轨迹

- `winning_solver_attempt_id`
- `winning_solver_answer`
- `winning_solver_reasoning`
- `winning_solver_thinking`
- `winning_solver_raw_model_text`
- `winning_solver_judge_result`
- `rewrite_source_trace`

其中：

- `winning_solver_*` 是最终被选中、用于后续改写的正确轨迹。
- `rewrite_source_trace` 是后续 SIP 重写的直接输入文本来源。

### 3.4 SIP 与自然语言推理

- `silver_sip_reasoning`
- `silver_quality_score`
- `silver_quality_tags`
- `silver_keep_as_reference`
- `natural_reasoning_paragraph`
- `raw_model_text_sip`

这里的 `SIP` 指 `Social Information Processing`。脚本里固定为四个阶段：

1. `cue_encoding`
2. `cue_interpretation`
3. `goal_clarification`
4. `response_generation`

也就是说，如果你要看“结构化社会推理骨架”，最关键的是：

- `silver_sip_reasoning.cue_encoding`
- `silver_sip_reasoning.cue_interpretation`
- `silver_sip_reasoning.goal_clarification`
- `silver_sip_reasoning.response_generation`
- `silver_sip_reasoning.natural_cot`

### 3.5 模型与 token 用量

- `usage_solver_winner`
- `usage_judge_winner`
- `usage_sip`
- `usage_natural`
- `model_name_solver`
- `model_name_judge`
- `model_name_sip`
- `model_name_natural`
- `worker_id`

## 4. `socialiqa_vote8_all_wrong.jsonl` 结构

`all_wrong` 的每一行对应一条“至少有可解析输出，但没有任何正确轨迹”的样本。顶层字段比 `raw` 更精简：

- `id`
- `story_id`
- `dimension`
- `order`
- `task_type`
- `full_story`
- `question`
- `correct_answer_text`
- `correct_answer_aliases`
- `wrong_answer_texts`
- `error`
- `num_parallel_solver_attempts`
- `num_parseable_solver_attempts`
- `num_correct_solver_attempts`
- `num_wrong_solver_attempts`
- `solver_attempts`
- `correct_solver_attempts`
- `wrong_solver_attempts`
- `solver_failures`
- `worker_id`

这个文件里最常见的 `error` 是：

- `no_correct_attempt_among_parseable_solver_calls`

它表示：

- 这些样本不是“解析失败”。
- 模型确实给出了可解析答案。
- 但 `vote8` 阶段没有任何一次命中 gold answer。

因此这个文件最适合用于：

- 失败案例分析
- 样本难度分析
- prompt / judge / model 调整
- self-consistency 覆盖率分析

