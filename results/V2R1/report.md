# LLM Judge 位置偏见实验报告

> 运行模式：真实 Judge API

> 候选回答版本：V2

## 指标汇总

| 条件 | Consistency | Flip Rate | Accuracy | Strong Win | Weak Win | Tie | Forced Tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Forward | — | — | 0.779 | 0.779 | 0.000 | 0.221 | — |
| Baseline Reverse | — | — | 0.735 | 0.735 | 0.000 | 0.265 | — |
| Baseline Overall | 0.868 | 0.132 | 0.757 | 0.757 | 0.000 | 0.243 | — |
| Swap-then-Merge | — | — | 0.691 | 0.691 | 0.000 | 0.309 | 0.132 |
| Reason-then-Judge Forward | — | — | 0.824 | 0.824 | 0.000 | 0.176 | — |
| Reason-then-Judge Reverse | — | — | 0.794 | 0.794 | 0.000 | 0.206 | — |
| Reason-then-Judge Overall | 0.853 | 0.147 | 0.809 | 0.809 | 0.000 | 0.191 | — |
| Reason-then-Judge + Swap-then-Merge | — | — | 0.735 | 0.735 | 0.000 | 0.265 | 0.147 |

## 解释模板

- 基线交换顺序后的翻转率为 0.132。明显高于 0 表明 Judge 对呈现位置敏感；应同时结合正序与逆序准确率差异判断偏好方向。
- Reason-then-Judge 的翻转率为 0.147。若低于基线且准确率不下降，则该提示干预有效。
- Swap-then-Merge 后强回答胜率为 0.691，强制平局率为 0.132。它以更多平局换取对冲突结论的保守处理。
- 两种方法结合后的强回答胜率为 0.735，强制平局率为 0.147。应与单独干预比较，而不能只看翻转率。
- Accuracy 按全部有效判决中选择 strong 的比例计算；tie 不计正确。题面公式中的第二项应为加号，而非减号。
