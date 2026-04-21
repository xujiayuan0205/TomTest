# ToMQA vote8 + SIP 推理输出说明

这个目录保存的是 `ToMQA` 数据在 `vote8` 并发求解流程上的输出结果。核心文件是：

- `tomqa_vote8_raw.jsonl`
- `tomqa_vote8_all_wrong.jsonl`

它们对应同一条产线的两类样本：

- `raw`：至少有一个并发 solver 尝试答对，因而可以继续做 `SIP` 重写。
- `all_wrong`：并发 solver 的可解析结果全部答错，没有可用于重写的正确链路。

按当前这次运行的 `run_summary.json`：

- 处理样本数：`500`
- `tomqa_vote8_raw.jsonl`：`366` 条
- `tomqa_vote8_all_wrong.jsonl`：`134` 条
- `tomqa_vote8_errors.jsonl`：`0` 条

也就是说，这次运行里 `raw + all_wrong = 500`，两者合起来覆盖了全部处理样本。

## 1. 整体流程

这套流程对应代码：

- `tomqa_vote8_parallel_launcher.py`
- `tomqa_vote8_parallel_worker.py`

单条样本的大致处理逻辑是：

1. 对每条 ToMQA 样本并发跑 `8` 次 solver 尝试。
2. 如果至少有一个可解析且正确的答案，这条样本进入 `tomqa_vote8_raw.jsonl`。
3. 选中一个正确 solver 的推理轨迹，重写成结构化 `SIP` 推理。
4. 再把结构化 `SIP` 重写成一个自然语言推理段落。
5. 如果所有可解析尝试都答错，这条样本进入 `tomqa_vote8_all_wrong.jsonl`。
6. 如果一个可解析结果都没有，才会进入 `tomqa_vote8_errors.jsonl`。

## 2. SIP 是什么

这里的 `SIP` 指的是 `Social Information Processing`。在脚本里，结构化重写被固定成四个阶段：

1. `Cue Encoding`
2. `Cue Interpretation`
3. `Goal Clarification`
4. `Response Generation`

对应到 `raw` 文件中的核心字段是：

- `silver_sip_reasoning.cue_encoding`
- `silver_sip_reasoning.cue_interpretation`
- `silver_sip_reasoning.goal_clarification`
- `silver_sip_reasoning.response_generation`
- `silver_sip_reasoning.natural_cot`

其中：

- `cue_encoding`：提取相关事实、社会线索、忽略项。
- `cue_interpretation`：建模角色的信念、意图、误解或隐藏状态。
- `goal_clarification`：明确当前问题下角色的心理指向和行动目标。
- `response_generation`：把前面三步收束到最终答案。
- `natural_cot`：基于 SIP 骨架生成的一段自然语言推理。

如果想看“ SIP 推理”本体，最关键的就是 `tomqa_vote8_raw.jsonl` 里的 `silver_sip_reasoning` 字段。

## 3. 两个文件的分工

| 文件 | 收录条件 | 是否包含 SIP 重写 | 主要用途 |
| --- | --- | --- | --- |
| `tomqa_vote8_raw.jsonl` | 至少一个 solver 尝试答对 | 是 | 保留成功样本、winning trace、SIP 骨架、自然推理段落 |
| `tomqa_vote8_all_wrong.jsonl` | 所有可解析 solver 尝试都答错 | 否 | 做失败案例分析、看 vote8 为何没覆盖到正确答案 |

要点是：

- `raw` 不是“所有原始尝试”，而是“成功进入 SIP 重写阶段的样本全集”。
- `all_wrong` 不是噪声文件，而是和 `raw` 互补的失败子集。
- `all_wrong` 里没有 `silver_sip_reasoning`，因为没有正确 solver 轨迹可供重写。

## 4. `tomqa_vote8_raw.jsonl` 结构

`raw` 的每一行是一个成功样本，顶层字段大致可以分为五组。

### 4.1 样本原始信息

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
- `state`
- `action`
- `meta`

### 4.2 vote8 求解统计

- `num_parallel_solver_attempts`
- `num_parseable_solver_attempts`
- `num_correct_solver_attempts`
- `num_wrong_solver_attempts`
- `solver_attempts`
- `correct_solver_attempts`
- `wrong_solver_attempts`
- `solver_failures`

这里的 `solver_attempts` 会保留每次 solver 尝试的详细信息，常见子字段包括：

- `attempt_id`
- `final_answer`
- `free_form_reasoning`
- `thinking`
- `raw_model_text`
- `parsed_by_fallback`
- `match_score`
- `is_correct`
- `usage`

### 4.3 winning solver 轨迹

- `winning_solver_attempt_id`
- `winning_solver_answer`
- `winning_solver_reasoning`
- `winning_solver_thinking`
- `winning_solver_raw_model_text`
- `rewrite_source_trace`

其中 `rewrite_source_trace` 是后续生成 SIP 的直接输入来源。

### 4.4 SIP 与自然语言推理

- `silver_sip_reasoning`
- `silver_quality_score`
- `silver_quality_tags`
- `silver_keep_as_reference`
- `natural_reasoning_paragraph`
- `raw_model_text_sip`

这部分就是后处理得到的“银标推理”：

- `silver_sip_reasoning`：结构化 SIP 骨架。
- `natural_reasoning_paragraph`：从 SIP 骨架改写出来的一段自然推理。
- `silver_quality_score` / `silver_quality_tags`：对 SIP 质量的启发式评估。
- `silver_keep_as_reference`：是否被保留进后续 `reference` / `sft` 数据。

### 4.5 模型与 token 用量

- `usage_solver_winner`
- `usage_sip`
- `usage_natural`
- `model_name_solver`
- `model_name_sip`
- `model_name_natural`
- `worker_id`

## 5. `tomqa_vote8_all_wrong.jsonl` 结构

`all_wrong` 的每一行对应一条“没有任何正确 solver 尝试”的样本。顶层字段更精简：

- `id`
- `story_id`
- `dimension`
- `order`
- `task_type`
- `full_story`
- `question`
- `correct_answer_text`
- `correct_answer_aliases`
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

这里通常会看到：

- `error = "no_correct_attempt_among_parseable_solver_calls"`

这说明：

- 这些样本不是解析失败。
- 模型确实给出了可解析答案。
- 但 `vote8` 的全部可解析尝试都没有命中 gold answer。

因此这个文件最适合做：

- 错误归因
- 样本难度分析
- prompt / model / self-consistency 策略调优

## 6. 两个文件怎么一起看

如果你把这两个文件放在一起理解，可以把它们看成同一个 `vote8 + SIP` 实验的完整切面：

- `tomqa_vote8_raw.jsonl`：成功面，回答了“哪些样本已经拿到了正确答案，并被成功转写为 SIP 推理？”
- `tomqa_vote8_all_wrong.jsonl`：失败面，回答了“哪些样本在 vote8 阶段就没过，因此根本没有进入 SIP 重写？”

一句话总结：

- `raw` 看的是“成功后的 SIP 推理长什么样”。
- `all_wrong` 看的是“哪些题连可用的正确推理源都没拿到”。

## 7. 和其它输出文件的关系

虽然这份 README 主要说明上面两个文件，但目录里另外几个文件和它们有直接关系：

- `run_summary.json`：本次运行的总体统计。
- `tomqa_vote8_reference.jsonl`：从 `raw` 里筛出 `silver_keep_as_reference = true` 的高质量样本。
- `tomqa_vote8_sft.jsonl`：把 `reference` 进一步转换成训练友好的 SFT 格式。
- `tomqa_vote8_errors.jsonl`：并发求解里没有得到可解析结果的样本。

所以这条链路实际上是：

`vote8 solver -> raw -> SIP rewrite -> reference -> sft`

而 `all_wrong` 是并行旁路，用来保存未通过 vote8 的失败样本。


