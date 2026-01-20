# Tasks

## AIME (American Invitational Mathematics Examination)

AIME 是美国数学邀请赛，是面向高中生的高难度数学竞赛。我们使用 AIME 2024 和 AIME 2025 的题目来评估模型的数学推理能力。

### 数据集

- **AIME-2024**: 来自 `math-ai/aime24`，包含 2024 年 AIME 竞赛题目
- **AIME-2025**: 来自 `math-ai/aime25`，包含 2025 年 AIME 竞赛题目

每道题目都是需要复杂推理的数学问题，答案是一个整数（0-999）。

### 评估方法

1. **生成式评估** (`eval_type = "generative"`)：模型需要生成完整的解题过程

2. **答案格式**：模型需要将最终答案用 `\boxed{...}` 标记，例如 `\boxed{42}`

3. **评分标准**：精确匹配 - 从模型输出中提取 `\boxed{...}` 内的内容，与标准答案进行字符串匹配

### Prompt 格式

```
Problem:
<问题内容>

Mark your solution with \boxed
Answer:
```

### 使用示例

```bash
# 评估 AIME-2024
python -m scripts.chat_eval -i sft -a "AIME-2024" -m 2048

# 评估 AIME-2025
python -m scripts.chat_eval -i sft -a "AIME-2025" -m 2048

# 同时评估两个年份
python -m scripts.chat_eval -i sft -a "AIME-2024|AIME-2025" -m 2048
```

**注意**：由于 AIME 题目需要较长的推理过程，建议使用 `-m 2048` 或更大的值来增加生成长度。

### 难度说明

AIME 是非常具有挑战性的数学竞赛，即使是大型语言模型在这个任务上的准确率也相对较低。这个评估主要用于测试模型的高级数学推理能力。

