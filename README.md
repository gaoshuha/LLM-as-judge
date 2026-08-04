# MT-Bench LLM Judge 位置偏见实验

## 1. 实验设计

`question.jsonl` 是标准 MT-Bench 的 80 条样本（ID 81–160），每条包含两个连续 turn。项目提供两套固定候选：

- `answers_v1.jsonl`：原始版本。`candidate_1` 事实正确、完整并遵循指令；`candidate_2` 可能有局部事实错误、遗漏或推理跳步，质量差距较大。
- `answers_v2.jsonl`：接近质量版本。`candidate_1` 与 V1 完全相同；`candidate_2` 从高质量回答做保守编辑，只删除非关键解释、示例、边界条件或增加轻微格式冗余，不主动引入错误。80 道题的两轮轨迹均至少有一处局部差异。

`answers.jsonl` 保留为 V1 的兼容副本。两套候选都只构造一次，正式实验不会动态生成。

Judge 每题执行四次盲评：Baseline 正序/逆序、Reason-then-Judge 正序/逆序。正序的 A/B 分别对应 candidate_1/candidate_2，逆序交换；发送内容只使用 `Response A`、`Response B`，不含真实映射。真实映射单独保存在 `ground_truth.json`，仅在 Judge 输出完成后用于离线统计。

映射回原候选后，位置一致率为

\[
\mathrm{Consistency}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat w_i^{f}=\hat w_i^{r}],
\]

翻转率为

\[
\mathrm{Flip\ Rate}=1-\mathrm{Consistency}.
\]

准确率按全部正逆序判决中选择高质量候选的比例计算：

\[
\mathrm{Accuracy}=\frac{1}{2N}\sum_{i=1}^{N}
\left(\mathbf{1}[\hat w_i^{f}=\mathrm{strong}]+\mathbf{1}[\hat w_i^{r}=\mathrm{strong}]\right).
\]

这里使用加号。任务说明中的减号与“选中高质量回答的比例”这一定义矛盾，会使逆序的正确判决产生负贡献。

Swap-then-Merge 在两个映射结果相同时保留该结果，否则输出 tie。Reason-then-Judge 要求先比较两边的优点和局限，再在 JSON 的 `final_winner` 字段给结论。

## 2. 数据格式

问题文件每行一条 JSON：

```json
{"question_id":81,"category":"writing","turns":["第一轮问题","第二轮问题"]}
```

回答文件不包含质量标签，每个候选是与两个问题 turn 对齐的数组：

```json
{"question_id":81,"candidate_1":["第一轮回答","第二轮回答"],"candidate_2":["第一轮回答","第二轮回答"]}
```

离线真实映射单独保存：

```json
{"question_id":81,"strong_candidate":"candidate_1","weak_candidate":"candidate_2","strong_answer_id":"candidate_1","weak_answer_id":"candidate_2"}
```

## 3. 程序与匿名化

主程序是 `position_bias_experiment.py`。Judge 的 user prompt 只呈现用户轮次和匿名的 `Response A` / `Response B`；不会读取或插入 `ground_truth.json` 的字段。API 调用固定 `temperature=0`、`seed=0`，使用 JSON mode，最多指数退避重试 5 次。每次成功调用立即追加保存，因此中断后再次运行会跳过已完成项目。

`--mock` 是确定性的管线测试器，不读取真实标签，也不代表任何 LLM 评测结论。

## 4. 运行方式

程序只使用 Python 标准库，无第三方依赖；建议 Python 3.10 或更新版本。

1. 打开 `position_bias_experiment.py` 顶部配置区，把 API Key 粘贴到 `JUDGE_API_KEY`。OpenAI 默认模型已配置为 `gpt-4.1-mini`。使用兼容服务时，同时修改 `JUDGE_MODEL_NAME` 和 `JUDGE_BASE_URL`。
2. 保持 `question.jsonl`、`answers_v1.jsonl`、`answers_v2.jsonl`、`ground_truth.json` 与脚本在同一目录。
3. 先做不调用 API 的小规模测试：

```powershell
python .\position_bias_experiment.py --answer-version 2 --mock --limit 3 --fresh
```

4. 正式执行全部 320 次 Judge 调用：

```powershell
python .\position_bias_experiment.py --answer-version 2 --fresh
```

省略 `--answer-version` 时，交互式终端会在任何 Judge 请求开始前显示菜单，要求选择：

```text
1 - V1：质量差距较大
2 - V2：质量接近，仅有次要遗漏或轻微格式冗余
```

在管道、任务调度器等非交互环境中必须显式传入 `--answer-version 1` 或 `--answer-version 2`，以免误用数据集。

若网络中断，不加 `--fresh` 重新执行即可断点续跑。V1 使用原文件 `judge_outputs.jsonl`、`metrics_summary.csv`、`report.md`；V2 使用独立的 `judge_outputs_v2.jsonl`、`metrics_summary_v2.csv`、`report_v2.md`。两个版本的断点与结果不会混用。`--fresh` 只清除当前所选版本的 prompt、输出和失败日志，不会删除候选回答或 ground truth。

运行真实 API 时，如果标准输出连接到真实控制台，会在同一位置刷新如下状态行：

```text
API状态 | 题目 90 | reason/forward | 等待首输出 | 38.0s | 第 1/5 次 | 输入处理≈0.0字符/s | 当前输出=0.0字符/s | 已收=0字符 | 按 S 跳过本题
```

- `等待首输出`：请求已发出，服务端尚未返回内容；此时速度显示为 0 是正常的，可能仍在排队或推理。
- `正在输出`：流式内容正在返回；“当前输出”是最近刷新区间的速度。
- 默认 30 秒没有任何首输出时，程序会强制终止该次调用、记录失败并继续下一个判决。以后断点续跑时会重试。可用 `--first-output-timeout 60` 修改，或设为 `0` 禁用。
- 运行中按 `S`：立即跳过当前整道题。每个 API 请求运行在独立子进程中；程序会终止并确认该进程已经退出，再开始下一题，不会留下占用连接的后台请求。
- 默认某一次 API 调用超过 300 秒会自动跳过当前整道题。可用 `--max-call-seconds 180` 修改，或设为 `0` 禁用自动跳过。

也可以在启动前指定跳过题号：

```powershell
python .\position_bias_experiment.py --skip-question 90
python .\position_bias_experiment.py --skip-question 90,105 --max-call-seconds 180
```

被跳过的题不会进入当次指标分母；跳过记录保留在 `judge_outputs.jsonl`。以后不带 `--skip-question` 重新运行时，程序会再次尝试该题，因此不会永久丢失样本。

为保证 Codex/IDE 日志面板绝不刷屏，代码顶部的 `ENABLE_LIVE_STATUS` 默认是 `False`：每次 API 调用只打印一条静态状态，不进行定时刷新。只有直接在支持光标控制的真实 Windows 控制台运行时，才可手动设为 `True`；程序还会再次用 Windows Console API 检查能力，不支持时仍自动回退为静态单行。

### 卡顿排查记录

早期断点记录中出现过用户连续跳过、远端连接重置和 HTTP 503。119–124 号题的 Baseline Prompt 只有约 692–2,692 个字符，并没有突然变成超长输入，因此卡顿不由题目长度直接造成。

旧实现使用后台线程包装阻塞式 `urlopen()`。按 `S` 后主流程会继续，但尚未从 `urlopen()` 返回的线程无法被 Python 立即强制停止；连续跳题可能留下多个同时占用连接的请求，进而增加服务端排队、连接重置和 503 的概率。现在改用可强制终止的独立子进程，并已通过本地测试确认自动跳过后不存在活动的 API 子进程。

进一步 API 诊断表明：模型列表接口在 1.32 秒返回，模型名存在；极短非流式请求 1.80 秒完成；极短流式请求 0.51 秒收到首块、1.47 秒完成；复现完整的 81 号 `baseline/reverse` Prompt 时，0.64 秒收到首输出、6.62 秒完成且 JSON 有效。因此 81 号并非内容固定触发卡顿，长时间无首输出属于服务端偶发排队或瞬时停滞。首输出硬超时用于隔离这种偶发请求。

运行后生成：`judge_prompts.jsonl`、`judge_outputs.jsonl`、`mapped_results.csv`、`metrics_summary.csv`、`report.md`；解析或请求失败另记入 `parse_failures.jsonl`。

## 5. 结果解释模板

- 基线 Flip Rate 明显高于 0，说明交换 A/B 后结论不稳定；再比较 Forward Accuracy 与 Reverse Accuracy，可以判断偏向先呈现还是后呈现的位置。
- Reason-then-Judge 若降低 Flip Rate 且 Accuracy 不下降，才可认为提示干预有效；只降低翻转率但稳定地选错并非改善。
- Swap-then-Merge 会把冲突保守地变为 tie。应同时报告合并后 Strong Win Rate、Accuracy 与 Forced Tie Rate，明确稳定性改善所付出的弃权代价。
- 两种方法结合应与基线、单独 Reason-then-Judge 和单独 Swap-then-Merge 同时比较。由于只有 80 题，正式论文建议对题目做 bootstrap 置信区间，并按八个 MT-Bench 类别分层报告。
- mock 结果只能证明文件、匿名化、解析与统计流程可以运行，不能用于判断 Judge 是否存在位置偏见。
