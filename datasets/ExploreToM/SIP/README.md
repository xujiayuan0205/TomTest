# ExploreToM Vote8 Outputs

这个目录保存了一次 `ExploreToM` vote-8 并行求解运行的合并结果，重点文件是：

- `exploretom_vote8_raw.jsonl`
- `exploretom_vote8_all_wrong.jsonl`

两者都是 `JSONL` 格式，每行一个样本对象，适合流式读取。

## 本次运行概况

信息来自 [run_summary.json](/d:/LLM_learn/TOM/ExploreToM/exploretom_vote8_outputs/run_summary.json)：

- 输入数据：[exploretom_550.json](/d:/LLM_learn/TOM/ExploreToM/exploretom_550.json)
- 样本范围：`[0, 500)`
- `num_workers=10`
- `num_parallel_attempts=8`
- `model_name_judge=qwen/qwen3-32b`
- `raw_rows=417`
- `all_wrong_rows=83`
- `error_rows=0`
- `reference_rows=158`
- `sft_rows=158`

因此，本次 500 条样本被分成了：

- `417` 条进入 `exploretom_vote8_raw.jsonl`
- `83` 条进入 `exploretom_vote8_all_wrong.jsonl`
- `0` 条进入 `exploretom_vote8_errors.jsonl`

## 两个主文件的区别

### `exploretom_vote8_raw.jsonl`

这个文件保存“至少有一个 solver attempt 被 judge 判为正确”的样本。

每条记录通常包含：

- 样本基本信息：`id`、`story_id`、`dimension`、`order`、`task_type`
- 原始题目内容：`full_story`、`story_summary`、`question`
- 标准答案：`correct_answer_text`、`correct_answer_aliases`
- 结构化状态：`state`、`action`、`meta`
- 并发求解统计：`num_parallel_solver_attempts`、`num_parseable_solver_attempts`、`num_correct_solver_attempts`、`num_wrong_solver_attempts`
- 全部尝试明细：`solver_attempts`、`correct_solver_attempts`、`wrong_solver_attempts`、`solver_failures`
- 胜出尝试信息：`winning_solver_attempt_id`、`winning_solver_answer`、`winning_solver_reasoning`、`winning_solver_thinking`、`winning_solver_raw_model_text`、`winning_solver_judge_result`
- 后处理结果：`rewrite_source_trace`、`silver_sip_reasoning`、`silver_quality_score`、`silver_quality_tags`、`silver_keep_as_reference`
- 自然语言推理段落：`natural_reasoning_paragraph`
- token/usage 信息：`usage_solver_winner`、`usage_judge_winner`、`usage_sip`、`usage_natural`
- 运行配置：`model_name_solver`、`model_name_judge`、`model_name_sip`、`model_name_natural`、`worker_id`

用途：

- 训练/蒸馏数据主来源
- 分析多次尝试中哪次被 judge 选中
- 回看 silver SIP rewrite 质量
- 从中筛选 `silver_keep_as_reference=true` 的高质量样本

### `exploretom_vote8_all_wrong.jsonl`

这个文件保存“模型有可解析输出，但 8 次并发尝试里没有任何一次被 judge 判为正确”的样本。

每条记录保留：

- 样本基本信息：`id`、`story_id`、`dimension`、`order`、`task_type`
- 原题内容：`full_story`、`question`
- 标准答案：`correct_answer_text`、`correct_answer_aliases`
- 失败标记：`error`
- 并发求解统计：`num_parallel_solver_attempts`、`num_parseable_solver_attempts`、`num_correct_solver_attempts`、`num_wrong_solver_attempts`
- 尝试明细：`solver_attempts`、`correct_solver_attempts`、`wrong_solver_attempts`、`solver_failures`
- `worker_id`

这里的 `error` 通常是：

- `no_correct_attempt_among_parseable_solver_calls`

也就是说，这不是“程序报错”，而是“模型答了，但都被判错”。

用途：

- 错题分析
- 检查 judge 是否过严
- 检查 parser/fallback 是否有脏解析
- 作为后续 prompt、parser、judge 调整时的回归测试集合

## 两个文件共享的关键字段

### `solver_attempts`

这是最重要的列表字段。每个元素对应一次 solver 调用，通常包含：

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

其中：

- `final_answer` 是当前 parser 最终抽取出来的答案
- `thinking` 是模型的 reasoning/thinking 内容
- `raw_model_text` 是最终可见文本输出
- `parsed_by_fallback=true` 表示没有按理想 JSON 路径解析成功，而是走了 fallback 提取
- `judge_result.is_equivalent=true` 表示该 attempt 被 judge 视为与 gold answer 语义等价

### `solver_failures`

这里记录的是“某次 solver 调用直接抛异常、没有形成可序列化 attempt”的情况。当前这次合并结果里 `exploretom_vote8_errors.jsonl` 为 `0` 行，因此主失败类型不是程序异常，而是语义判错。

## 与其他文件的关系

- [exploretom_vote8_reference.jsonl](/d:/LLM_learn/TOM/ExploreToM/exploretom_vote8_outputs/exploretom_vote8_reference.jsonl)
  从 `exploretom_vote8_raw.jsonl` 中筛出 `silver_keep_as_reference=true` 的样本，本次共 `158` 条。
- [exploretom_vote8_sft.jsonl](/d:/LLM_learn/TOM/ExploreToM/exploretom_vote8_outputs/exploretom_vote8_sft.jsonl)
  由 `reference.jsonl` 进一步转换得到的训练格式文件。
- [exploretom_vote8_errors.jsonl](/d:/LLM_learn/TOM/ExploreToM/exploretom_vote8_outputs/exploretom_vote8_errors.jsonl)
  保存真正的运行异常样本，本次为空。
- `workers/`
  保存各个 worker 的分片输出，顶层的 `raw/all_wrong/reference/errors` 都是从这些分片合并来的。

