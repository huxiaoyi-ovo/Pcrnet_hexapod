# 项目总览（PROJECT_OVERVIEW_CN）

## 1) 项目目标

- 在单卡 3090 吞吐约束下训练具备泛化与迁移能力的六足策略，形成可复现、能支撑 RAL 论文结论的证据。完整长期目标与训练/部署口径见[实验规则](docs/agents/EXPERIMENT_RULES_CN.md)。

## 2) 技术背景与决策优先级

- `技术方案/hexapod_RAL_complete_technical_spec_v7.md` 与 `技术方案/hexapod_RAL_integrated_final_v7.md` 是阶段性技术背景。后续有日期且适用范围明确的实验决策优先；按当前代码、配置和记录确认实现与验收状态。
- 协作计划、执行授权、角色分工及专项读取条件以根目录 `AGENTS.md` 为准。返修、实验、训练一致性审计与交接细则按任务读取 `docs/agents/` 中相应文件。

## 3) 工程边界

- 核心训练与环境代码：`legged_gym/`、`rsl_rl/`。
- 资源与工具：`resources/`、`tools/`。
- 文档治理默认不改动以上目录的核心实现文件。

## 4) 文档职责

- `AGENTS.md`：常驻协作规则和专项入口。
- `docs/agents/REVISION_GUIDE_CN.md`：返修证据链和表述边界。
- `docs/agents/EXPERIMENT_RULES_CN.md`：实验、传感、场景与部署输入口径。
- `docs/agents/TRAINING_AUDIT_CN.md`：训练一致性与奖励生命周期审计。
- `docs/agents/HANDOFF_TEMPLATE_CN.md`：按需生成的会话交接模板。
- `TODO_LOG.md`：重大改变、方向调整与里程碑。
- `DEBUG_SUMMARY_CN.md`：调试总结；使用前核实当前内容。
- `docs/README.md`、`docs/NAVIGATION.md`：文档索引与快速导航。

`CONTEXT_HANDOFF_SUMMARY.md` 是按需生成的交接记录名称，不是当前仓库中的必读文件。

## 5) 想法（仅在用户明确要求“记忆想法”时维护）

- 暂无。
