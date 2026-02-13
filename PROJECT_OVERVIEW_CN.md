# 项目总览（PROJECT_OVERVIEW_CN）

## 1) 项目目标

- 在单卡 3090 的吞吐约束下，训练具备泛化/迁移能力的六足策略，并形成可复现、可支撑 RAL 论文的证据链。

## 2) 方案基准（唯一）

- `技术方案/hexapod_RAL_complete_technical_spec_v7.md`
- `技术方案/hexapod_RAL_integrated_final_v7.md`

## 3) 工程边界

- 核心训练与环境代码：`legged_gym/`、`rsl_rl/`
- 资源与工具：`resources/`、`tools/`
- 文档与流程优化默认不改动以上目录的核心实现文件。

## 4) 文档职责矩阵

- `TODO_LOG.md`：执行与里程碑（仅重大改变/思路调整）。
- `DEBUG_SUMMARY_CN.md`：调试总结与稳定口径。
- `CONTEXT_HANDOFF_SUMMARY.md`：会话交接摘要。
- `docs/README.md`：文档结构总索引。
- `docs/NAVIGATION.md`：快速导航（当前规范/当前参考/历史参考）。

## 5) 想法（仅在用户明确要求“记忆想法”时维护）

- 暂无。
