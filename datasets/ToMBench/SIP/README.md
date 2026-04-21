# TomBench Vote8 Outputs

这个目录里最重要的两个文件是：

- `tombench_vote8_raw.jsonl`
- `tombench_vote8_all_wrong.jsonl`

它们都来自同一个 `vote8` 流程，但含义不同。

## 这两个文件分别是什么

- `tombench_vote8_raw.jsonl`
  表示样本经过 `8` 次并行 solver 尝试后，至少有可用结果进入了后续重写流程，并最终产出了：
  - winning solver
  - silver SIP 推理链
  - 自然语言推理段落

- `tombench_vote8_all_wrong.jsonl`
  表示样本在并行 solver 阶段 `8` 次尝试全部没有答对，因此没有进入后续 SIP 重写阶段。
  这类样本本质上是：
  - vote8 流程中的“难例”或“失败例”
  - 可用于分析 solver 在哪些题型上集体失误

一句话理解：

- `raw` = 进入了后续 SIP 重写的成功处理样本
- `all_wrong` = 在前面的 8 次 solver 投票里全错的样本

## 共同字段

这两个文件里都常见的字段有：

- `id`
  样本唯一 ID，通常格式类似 `story_0000003__qa1`。
- `story_id`
  原始故事 ID。
- `qa_index`
  该故事下的问题编号。
- `ability`
  TomBench 原始能力标签，例如 `Hidden emotions`、`Faux pas`、`Second-order beliefs`。
- `full_story`
  故事全文。
- `question`
  解析后的题干。
- `question_raw`
  原始多选题文本，保留了选项串。
- `options`
  选项字典，例如 `{"A": "...", "B": "...", "C": "...", "D": "..."}`。
- `correct_answer_label`
  标准答案字母，例如 `B`。
- `correct_answer_text`
  标准答案文本，例如 `Embarrassment`。
- `correct_answer_full`
  标准答案完整格式，例如 `B. Embarrassment`。

## Vote8 投票字段

这两个文件都带有 vote8 相关统计：

- `num_parallel_solver_attempts`
  并行 solver 尝试次数。通常是 `8`。
- `num_correct_solver_attempts`
  8 次 solver 中答对的次数。
- `num_wrong_solver_attempts`
  8 次 solver 中答错的次数。
- `solver_attempts`
  所有 solver 尝试的完整列表。
- `correct_solver_attempts`
  从 `solver_attempts` 中筛出的答对尝试。
- `wrong_solver_attempts`
  从 `solver_attempts` 中筛出的答错尝试。
- `solver_failures`
  solver 调用失败或解析失败的记录。
- `worker_id`
  并行运行时产生这条记录的 worker 编号。

## 单个 `solver_attempt` 表示什么

`solver_attempts` 中每一项通常包含：

- `attempt_id`
  第几次 solver 尝试。
- `selected_option_label`
  该次 solver 选择的选项字母。
- `selected_option_text`
  该次 solver 选择的选项文本。
- `selected_option_full`
  完整选项格式，例如 `B. Embarrassment`。
- `free_form_reasoning`
  solver 给出的简洁推理。
- `thinking`
  solver 的较长推理文本。
- `raw_model_text`
  solver 原始输出。
- `match_score`
  内部匹配得分。
- `is_correct`
  该次 solver 是否答对。
- `usage`
  这次调用的 token 使用信息。

## `tombench_vote8_raw.jsonl` 额外包含什么

`raw` 文件比 `all_wrong` 多出后续重写阶段的信息，常见字段包括：

- `state`
  原始样本中的状态信息。
- `action`
  原始样本中的动作信息。
- `meta`
  原始样本中的元信息。
- `winning_solver_attempt_id`
  被选中的 winning solver 编号。
- `winning_solver_answer`
  winning solver 的最终答案对象，通常包含：
  `selected_option_label`, `selected_option_text`, `selected_option_full`
- `winning_solver_free_form_reasoning`
  winning solver 的自由推理。
- `winning_solver_thinking`
  winning solver 的长推理文本。
- `winning_solver_raw_model_text`
  winning solver 的原始模型输出。
- `rewrite_source_trace`
  后续 SIP 重写所依赖的来源推理文本。
- `silver_sip_reasoning`
  结构化 SIP 推理链。
- `silver_quality_score`
  自动质量分数，通常可按 `0-100` 理解。
- `silver_quality_tags`
  自动质量诊断标签。
- `silver_keep_as_reference`
  是否保留为较高质量参考样本。
- `natural_reasoning_paragraph`
  基于 SIP 骨架改写出的自然语言推理段落。
- `usage_solver_winner`
  winning solver 阶段 token 使用信息。
- `usage_sip`
  SIP 生成阶段 token 使用信息。
- `usage_natural`
  自然推理重写阶段 token 使用信息。
- `raw_model_text_sip`
  SIP 阶段模型原始输出。
- `model_name_solver`
  solver 阶段使用的模型名。
- `model_name_sip`
  SIP 阶段使用的模型名。
- `model_name_natural`
  自然推理重写阶段使用的模型名。

## `tombench_vote8_all_wrong.jsonl` 额外包含什么

`all_wrong` 文件保留的是 solver 阶段全错的样本，因此没有 winning solver 重写结果，也没有 SIP 结构化输出。

它的关键额外字段是：

- `error`
  失败原因。当前这类数据里常见值是：
  `no_correct_attempt_among_parallel_solver_calls`

这表示：

- 一共做了 8 次并行 solver 尝试
- 没有一次选中了 gold answer
- 因此该样本没有进入后续 SIP 重写流程

## 如何理解这两个文件的关系

你可以把它们看成同一个 vote8 流程的两个分支输出：

- `tombench_vote8_raw.jsonl`
  处理成功并进入后续重写链的样本
- `tombench_vote8_all_wrong.jsonl`
  在最前面的 solver 阶段就“全错退出”的样本

如果你想分析：

- 哪些题模型至少能在 8 次里碰到正确方向
  主要看 `raw`
- 哪些题 8 次都做不对
  主要看 `all_wrong`

## `ability` 是什么标签

`ability` 是 TomBench 原始任务标签，表示题目主要考察的社会推理能力，例如：

- `Hidden emotions`
- `Belief-based action/emotion`
- `Content false beliefs`
- `Faux pas`
- `Discrepant emotions`
- `Emotion regulation`
- `Second-order beliefs`
- `White lies`
- `Information-knowledge links`

它是任务类型标签，不是质量标签。

## `silver_quality_tags` 是什么

只有 `tombench_vote8_raw.jsonl` 才会有 `silver_quality_tags`，因为只有这部分数据进入了 SIP 重写和质量评估。

这些标签表示自动规则检测到的潜在问题，例如：

- `encoding_impure`
- `exact_answer_leak_early`
- `strong_answer_overlap_early`
- `moderate_answer_overlap_early`
- `answer_direction_too_close_to_surface_answer`
- `goal_stage_generic`
- `natural_cot_missing_exact_answer`

它们用于：

- 数据清洗
- 自动筛选
- 误差分析

不建议把它们当成 gold label 直接训练。

## 建议怎么用

- 如果你要做 SIP 监督训练、重写分析、质量筛选：
  优先用 `tombench_vote8_raw.jsonl`

- 如果你要做 hard-case 分析、solver 失败模式研究、题型难度分析：
  优先用 `tombench_vote8_all_wrong.jsonl`

- 如果你要研究 vote8 多次采样的一致性：
  两个文件都值得一起看，重点字段是：
  `solver_attempts`, `num_correct_solver_attempts`, `num_wrong_solver_attempts`

