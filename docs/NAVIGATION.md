# 文档导航

先读根目录 [`AGENTS.md`](../AGENTS.md) 获取协作规则和按需读取条件；按任务读取专项指南，不要求每次打开全部文档。

## 按任务查找

- 返修、回复信或审稿意见：[`agents/REVISION_GUIDE_CN.md`](agents/REVISION_GUIDE_CN.md)
- 训练、评测、场景、传感或部署输入的相关讨论、设计、分析与修改：[`agents/EXPERIMENT_RULES_CN.md`](agents/EXPERIMENT_RULES_CN.md)
- 训练一致性审计，或 reward/termination/reset/logging/门控/成功定义：[`agents/TRAINING_AUDIT_CN.md`](agents/TRAINING_AUDIT_CN.md)
- 用户要求“主动总结”：[`agents/HANDOFF_TEMPLATE_CN.md`](agents/HANDOFF_TEMPLATE_CN.md)
- 重大改变和里程碑：[`../TODO_LOG.md`](../TODO_LOG.md)
- 调试总结：[`../DEBUG_SUMMARY_CN.md`](../DEBUG_SUMMARY_CN.md)，使用前核实内容。
- 项目背景和文档职责：[`../PROJECT_OVERVIEW_CN.md`](../PROJECT_OVERVIEW_CN.md)

复合任务按需组合读取。与专项规则无关的普通问候、解释和单步状态查询不需要专项指南。

## 技术与实验资料

- V7 阶段技术背景：`../技术方案/hexapod_RAL_complete_technical_spec_v7.md`、`../技术方案/hexapod_RAL_integrated_final_v7.md`。后续有日期实验决策在各自适用范围优先。
- PCR 当前具体 `w` 计划：[PCR W 主线计划](specs/PCR_W_MAINLINE_PLAN_CN.md)；执行前按冻结协议、公式名和 checkpoint 识别版本。
- PCR RAL 协议：`specs/PCR_RAL_DAY1_CLAIM_EVAL_PROTOCOL_CN.md`、`specs/PCR_THEORY_DEFINITIONS_CN.md`。
- 操作手册：`operations/训练指令.txt`、`operations/PHASE_SWITCHING_GUIDE.md`。
- 参数参考：`reference/参数一览表.md`、`reference/ROBOT_SPECS.md`。
- RAL 自查与计划：`reference/论文写作自查清单.pdf`、`reference/RAL_PCR_TWO_WEEK_PLAN_CN.md`。
- 历史材料：`archive/思路设计.md`，仅用于追溯。

`CONTEXT_HANDOFF_SUMMARY.md` 是按需生成的交接记录名称，当前仓库没有该文件。

## 维护

新文档先确定归属；路径调整时同步更新本页、`README.md` 和 `docs/README.md`。文档治理默认不改 `legged_gym/`、`rsl_rl/`、`resources/`、`tools/`。
