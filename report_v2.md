# LLM Judge 位置偏见实验报告

> 运行模式：真实 Judge API

> 候选回答版本：V2

## 指标汇总

| 条件 | Consistency | Flip Rate | Accuracy | Strong Win | Weak Win | Tie | Forced Tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Forward | — | — | 0.762 | 0.762 | 0.000 | 0.237 | — |
| Baseline Reverse | — | — | 0.688 | 0.688 | 0.000 | 0.312 | — |
| Baseline Overall | 0.700 | 0.300 | 0.725 | 0.725 | 0.000 | 0.275 | — |
| Swap-then-Merge | — | — | 0.575 | 0.575 | 0.000 | 0.425 | 0.300 |
| Reason-then-Judge Forward | — | — | 0.863 | 0.863 | 0.000 | 0.138 | — |
| Reason-then-Judge Reverse | — | — | 0.787 | 0.787 | 0.000 | 0.212 | — |
| Reason-then-Judge Overall | 0.775 | 0.225 | 0.825 | 0.825 | 0.000 | 0.175 | — |
| Reason-then-Judge + Swap-then-Merge | — | — | 0.713 | 0.713 | 0.000 | 0.287 | 0.225 |

## 解释模板

- 基线交换顺序后的翻转率为 0.300。明显高于 0 表明 Judge 对呈现位置敏感；应同时结合正序与逆序准确率差异判断偏好方向。
- Reason-then-Judge 的翻转率为 0.225。若低于基线且准确率不下降，则该提示干预有效。
- Swap-then-Merge 后强回答胜率为 0.575，强制平局率为 0.300。它以更多平局换取对冲突结论的保守处理。
- 两种方法结合后的强回答胜率为 0.713，强制平局率为 0.225。应与单独干预比较，而不能只看翻转率。
- Accuracy 按全部有效判决中选择 strong 的比例计算；tie 不计正确。题面公式中的第二项应为加号，而非减号。
