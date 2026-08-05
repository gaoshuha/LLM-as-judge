# LLM Judge 位置偏见实验报告

> 运行模式：真实 Judge API

> 候选回答版本：V2

## 指标汇总

| 条件 | Consistency | Flip Rate | Accuracy | Strong Win | Weak Win | Tie | Forced Tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline Forward | — | — | 0.781 | 0.781 | 0.000 | 0.219 | — |
| Baseline Reverse | — | — | 0.672 | 0.672 | 0.000 | 0.328 | — |
| Baseline Overall | 0.859 | 0.141 | 0.727 | 0.727 | 0.000 | 0.273 | — |
| Swap-then-Merge | — | — | 0.656 | 0.656 | 0.000 | 0.344 | 0.141 |
| Reason-then-Judge Forward | — | — | 0.781 | 0.781 | 0.000 | 0.219 | — |
| Reason-then-Judge Reverse | — | — | 0.734 | 0.734 | 0.000 | 0.266 | — |
| Reason-then-Judge Overall | 0.797 | 0.203 | 0.758 | 0.758 | 0.000 | 0.242 | — |
| Reason-then-Judge + Swap-then-Merge | — | — | 0.656 | 0.656 | 0.000 | 0.344 | 0.203 |

## 解释模板

- 基线交换顺序后的翻转率为 0.141。明显高于 0 表明 Judge 对呈现位置敏感；应同时结合正序与逆序准确率差异判断偏好方向。
- Reason-then-Judge 的翻转率为 0.203。若低于基线且准确率不下降，则该提示干预有效。
- Swap-then-Merge 后强回答胜率为 0.656，强制平局率为 0.141。它以更多平局换取对冲突结论的保守处理。
- 两种方法结合后的强回答胜率为 0.656，强制平局率为 0.203。应与单独干预比较，而不能只看翻转率。
- Accuracy 按全部有效判决中选择 strong 的比例计算；tie 不计正确。题面公式中的第二项应为加号，而非减号。
