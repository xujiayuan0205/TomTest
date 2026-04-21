# ToMi 输出说明

raw/ 运行8次至少一次正确的样例 454
all_wrong/ 8次全错的样例 46

## 数据结构说明

每个样例核心字段结构如下：

*   
    "State" 等原始字段保留

    "Belief_Trace_by_DelTom" （**新增字段**）: {
        "input": {
            "sample_id": str,
            "story":     str,
            "question":  str,
            "answer":    str
        },
        "output": {
            "best_traces": List[str]   # *质量最好的trace，用于训练*,
            "all_correct": bool        # *是否至少一个答对* 
            "num_traces": int,
            "responses": [
                {
                    "attempt_id":         int,
                    "raw_model_response": str, # 原始输出
                }
            ]
        }
    }