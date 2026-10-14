# 训练一致性审计规则

用户要求“训练一致性审计”“一致性 review”或“审计式 review”，或修改 reward、termination、reset、logging、门控/成功定义时读取。目标同时识别当前实例与可重复问题类别，不止报单个 bug。先按当前代码和数据定位，不编造训练语义或影响。

## 输出结构

结果固定分四部分：`Findings`、`Patterns`、`AGENTS Update Proposal`、`TODO Suggestion`。

### Findings：Instance-level issue

每项写清：位置（文件/函数/分支）、触发条件、如何污染训练/评测/日志、影响 `actor / critic / entropy / GAE / bootstrap / logging` 的哪些部分、严重性 `P0 / P1 / P2`。只列有证据的问题；没有问题时明确说明范围与证据边界。

### Patterns：Pattern-level issue

若实例属于可复发类别，写类别名、一句话定义、典型症状、常见传播路径、为何普通训练现象难以发现、后续检查与防止方法。区分已观察事实与推测，不为单点故障强行归类。

### AGENTS Update Proposal

只在问题高概率重复出现、涉及训练口径一致性、会静默污染训练、易在其他分支复发，或适合形成长期规则/检查清单时建议新增规则。局部变量写错、一次性兼容、纯参数、不会扩散的简单问题不新增；需明确写：`AGENTS: 不新增，原因是___`。长期规则优先写入本专项文件，根 `AGENTS.md` 保留入口，避免静默扩根。

若建议新增，使用格式：

```markdown
## [规则标题]
- 适用范围：
- 风险描述：
- 强制规则：
- 最小检查项：
- 禁止事项：
```

### TODO Suggestion

仅列最小后续动作：先修什么、先加什么断言、先补什么检查，以及哪些必须在继续训练前解决。

## 优先检查的四类问题

1. **样本语义不一致**：`invalid / clamp / expert takeover / timeout` 在 `actor / critic / entropy / GAE / bootstrap / logging` 中处理不一致。
2. **观测语义不一致**：actor/critic 输入漂移，或 `goal_buf / local_map / difficulty / risk` 在训练、后处理、评测中的定义不一致。
3. **统计口径不一致**：`train / eval / play` 的 `success / progress / collision / stage` 定义不一致，或统计窗口混入旧样本。
4. **梯度与生命周期问题**：`detach / no_grad` 位置错误、rollout 保留不必要计算图、缓存跨 iteration 积累、更新阶段显存异常增长。

## 奖励生命周期与门控语义审计

对 reward、termination、reset、logging 相关改动额外执行本节，尤其是高层 `avoid / follow / moe / pcr` 中新增或修改奖励、终止、门控和成功判据后。

- **风险**：代码能运行、日志看似正常，但奖励被覆盖，门控参考量错误，观测成为死信号，或 done/reset 前后混入跨回合语义，静默污染结论。
- **必须核对**：每个 reward 属于加法、覆盖、清零还是终止改写；每个 gate/mask 的参考量是否符合任务语义；日志宣称生效的项是否进入 `total`；success/fail 是否与最终奖励一致。
- **最小检查项**：`collision/terminal` 是否语义反转；`near/far` 是否用错参考量；`goal_buf/local_map/passable_dir` 是否仍承载奖励实际使用的信息；done 步是否清零 shaped reward 且不污染下一回合缓存；`train/play/eval/best` 是否用同一语义解释指标。
- **禁止事项**：未核对覆盖/清零关系前直接调 reward 系数；未核对 gate 参考量前仅凭行为解释训练趋势；把长期为零或恒定的量继续当成有效奖励信号。
