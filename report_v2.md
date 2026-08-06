# LLM Judge 位置偏见实验报告

> 运行模式：MOCK（仅验证流程，不可作为实验结论）

> 候选回答版本：V2

## 指标汇总

| 条件 | Consistency | Flip Rate | Accuracy | Strong Win | Weak Win | Tie | Forced Tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Forward | — | — | 0.500 | 0.500 | 0.500 | 0.000 | — |
| Baseline Reverse | — | — | 0.500 | 0.500 | 0.500 | 0.000 | — |
| Baseline Overall | 0.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.000 | — |
| Swap-then-Merge | — | — | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| Reason-then-Judge Forward | — | — | 1.000 | 1.000 | 0.000 | 0.000 | — |
| Reason-then-Judge Reverse | — | — | 0.000 | 0.000 | 1.000 | 0.000 | — |
| Reason-then-Judge Overall | 0.000 | 1.000 | 0.500 | 0.500 | 0.500 | 0.000 | — |
| Reason-then-Judge + Swap-then-Merge | — | — | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

## 解释模板

- 基线交换顺序后的翻转率为 1.000。明显高于 0 表明 Judge 对呈现位置敏感；应同时结合正序与逆序准确率差异判断偏好方向。
- Reason-then-Judge 的翻转率为 1.000。若低于基线且准确率不下降，则该提示干预有效。
- Swap-then-Merge 后强回答胜率为 0.000，强制平局率为 1.000。它以更多平局换取对冲突结论的保守处理。
- 两种方法结合后的强回答胜率为 0.000，强制平局率为 1.000。应与单独干预比较，而不能只看翻转率。
- Accuracy 按全部有效判决中选择 strong 的比例计算；tie 不计正确。题面公式中的第二项应为加号，而非减号。
