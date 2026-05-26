# 文档体系（Architecture Docs Map）

本目录用于承载项目架构优化后的文档分层，目标是降低重复、明确职责、提升交接效率。

## 职责矩阵（Single Source of Truth）

- `技术方案/hexapod_RAL_complete_technical_spec_v7.md`、`技术方案/hexapod_RAL_integrated_final_v7.md`：技术方案唯一基准（V7）。
- `TODO_LOG.md`：执行与里程碑记录（仅重大改变/思路调整）。
- `DEBUG_SUMMARY_CN.md`：稳定调试经验与排障口径。
- `CONTEXT_HANDOFF_SUMMARY.md`：会话交接摘要模板与最新交接内容。
- `PROJECT_OVERVIEW_CN.md`：项目长期背景、范围边界与文档导航。
- `docs/NAVIGATION.md`：文档快速导航（当前规范 / 当前参考 / 历史参考）。

## 目录分层

- `docs/specs/`：规范与方案索引。
- `docs/operations/`：训练/评测/运行操作手册。
- `docs/reference/`：参数口径、术语、稳定参考。
- `docs/archive/`：历史草案与阶段性记录（仅参考，不作为当前规范）。

## 已归位文档（第二阶段）

- `docs/specs/PCR_RAL_DAY1_CLAIM_EVAL_PROTOCOL_CN.md`
- `docs/specs/PCR_W_MAINLINE_PLAN_CN.md`
- `docs/specs/PCR_THEORY_DEFINITIONS_CN.md`
- `docs/operations/训练指令.txt`
- `docs/operations/PHASE_SWITCHING_GUIDE.md`
- `docs/reference/RAL_PCR_TWO_WEEK_PLAN_CN.md`
- `docs/reference/论文写作自查清单.pdf`
- `docs/reference/参数一览表.md`
- `docs/reference/ROBOT_SPECS.md`
- `docs/archive/思路设计.md`

## 当前迁移策略

当前采用“骨架已建 + 分批归位”：

- 先创建目录和索引，不触碰核心代码目录。
- 已迁移文档优先在 `docs/` 维护；根目录仅保留核心入口文档。
