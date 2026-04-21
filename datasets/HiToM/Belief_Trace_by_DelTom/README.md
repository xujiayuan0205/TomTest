# HiToM 输出说明

HiToM本身就提供了belief trace数据，可直接使用，故没有all_wrong，该文件夹为字段对齐后的结果：

raw/ 正确样例

## 数据结构说明

每个样例核心字段结构如下：

*   
    "State" 等原始字段保留

    "Belief_Trace_by_DelTom" （**新增字段**）: {
        "output": {
            "best_traces": List[str]   # *质量最好的trace，用于训练*
        }
    }