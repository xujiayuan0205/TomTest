# ToMi Vote-8 Outputs

本目录中最终建议对外使用的两个核心文件是：

- `tomi_vote8_raw.jsonl`
- `tomi_vote8_all_wrong.jsonl`


## 文件作用

### `tomi_vote8_raw.jsonl`

这个文件保存的是“模型至少有一个可解析且判定正确的回答”的样本。

每条样本的大致流程是：

1. 给大模型输入 `full_story` 和 `question`
2. 并行采样多次 solver 回答
3. 将 solver 的 `final_answer` 与标准答案 `correct_answer_text` / `correct_answer_aliases` 进行比较
4. 只要存在至少一个正确回答，就从正确回答中选出一个 `winner`
5. 基于该 `winner` 的推理轨迹继续生成 SIP 改写和自然语言推理段落

因此，`tomi_vote8_raw.jsonl` 可以理解为“成功保留下来的可用监督数据”。

### `tomi_vote8_all_wrong.jsonl`

这个文件保存的是“没有任何可解析且判定正确的 solver 回答”的样本。

这些样本没有进入后续 SIP 改写阶段，通常用于：

- 分析模型在哪些题型上容易出错
- 统计整体成功率
- 后续做错误分析、prompt 调整或模型替换

## 两个文件的关系

- `tomi_vote8_raw.jsonl` 和 `tomi_vote8_all_wrong.jsonl` 是互补关系
- 同一个样本只会出现在其中一个文件里
- `raw` 表示本题至少找到了一条正确 solver 轨迹
- `all_wrong` 表示本题所有可解析 solver 轨迹都未答对，或者没有正确轨迹可用于后续 SIP 生成

## 顶层字段说明

两个文件都包含一部分共同字段：

- `id`, `story_id`: 样本编号
- `dimension`, `order`, `task_type`: 题型与元信息
- `full_story`, `question`: 输入给 solver 的故事和问题
- `correct_answer_text`, `correct_answer_aliases`: 标准答案及其可接受别名
- `num_parallel_solver_attempts`: 并行采样次数
- `num_parseable_solver_attempts`: 能成功解析出合法 `final_answer` 的次数
- `num_correct_solver_attempts`: 判定正确的次数
- `num_wrong_solver_attempts`: 判定错误的次数
- `solver_attempts`: 所有可解析 solver 回答
- `correct_solver_attempts`: 被判定为正确的 solver 回答子集
- `wrong_solver_attempts`: 被判定为错误的 solver 回答子集
- `solver_failures`: 未成功解析或调用失败的 attempt 记录
- `worker_id`: 生成该样本的 worker 编号

### `tomi_vote8_raw.jsonl` 独有字段

- `state`, `action`, `meta`: 原始数据中的结构化上下文
- `winning_solver_attempt_id`: 被选中的正确 solver attempt
- `winning_solver_answer`: 被选中 attempt 的最终答案
- `winning_solver_reasoning`: 被选中 attempt 的显式 reasoning 字段
- `winning_solver_thinking`: 被选中 attempt 的中间推理轨迹
- `winning_solver_raw_model_text`: 被选中 attempt 的原始模型文本
- `rewrite_source_trace`: 用于后续 SIP 改写的源推理文本
- `silver_sip_reasoning`: 基于正确轨迹生成的 SIP 结构化推理
- `silver_quality_score`, `silver_quality_tags`, `silver_keep_as_reference`: SIP 质量评估结果
- `natural_reasoning_paragraph`: 将 SIP 重写后的自然语言推理段落
- `usage_solver_winner`, `usage_sip`, `usage_natural`: 各阶段 token 使用信息
- `raw_model_text_sip`: SIP 阶段原始输出
- `model_name_solver`, `model_name_sip`, `model_name_natural`: 各阶段模型名

### `tomi_vote8_all_wrong.jsonl` 特有字段

- `error`: 当前样本进入 `all_wrong` 的原因

常见值如：

- `no_correct_attempt_among_parseable_solver_calls`

这表示该样本有可解析回答，但没有任何一个回答被判为正确。

## `solver_attempts` 中常见字段

`solver_attempts` 是一个列表，每个元素表示一次 solver 尝试。常见字段包括：

- `attempt_id`: 第几次采样
- `final_answer`: 解析后的最终答案，通常是标准化后的单个 token
- `free_form_reasoning`: 从模型输出中提取到的显式 reasoning
- `thinking`: 模型的中间推理文本
- `raw_model_text`: 原始模型回复
- `parsed_by_fallback`: 是否通过兜底规则解析出答案
- `match_score`: 与标准答案的匹配分数
- `is_correct`: 是否被判为正确
- `usage`: 该次调用的 token 使用统计

## 使用建议

- 如果你要构建 SIP 监督数据，优先使用 `tomi_vote8_raw.jsonl`
- 如果你要做失败分析、题型分布分析或 prompt 调整，查看 `tomi_vote8_all_wrong.jsonl`
- 若按行读取，建议使用流式 JSONL 读取方式，而不是一次性全部加载到内存

## 当前运行摘要

根据本次运行生成的汇总信息：

- 输入数据源：`tomi_550.json`
- 处理区间：`start_index=100`, `end_index=500`
- 最大样本数：`400`
- worker 数量：`10`
- 每题并行 solver 次数：`8`
- `tomi_vote8_raw.jsonl` 当前共有 `402` 行
- `tomi_vote8_all_wrong.jsonl` 当前共有 `98` 行

## 备注

- `tomi_vote8_raw.jsonl` 中的 `thinking`、`winning_solver_thinking` 和 `rewrite_source_trace` 可能包含较长的中间推理文本
- `final_answer` 已按数据集的 canonical token 风格标准化，便于与 `correct_answer_text` 和 `correct_answer_aliases` 对齐
- 若只上传这两个文件，本文档即可作为最小说明文件一起提供
