# Context Handoff Summary 模板

用户说“主动总结”时，生成可直接用于新会话的交接摘要。只保留长期有用的目标、约束、决策、现状、风险与下一步；删去闲聊和推导过程。提到文件、模块或 TODO 时写清路径并使用可执行动词。缺失或不确定的信息标为 `【未知】`，不要编造。

长度建议：400–900 中文字或 200–450 英文词。标题、章节和字段保持以下结构，不增加章节；第 2、8 节按字段和待办列表排版。

# Context Handoff Summary

## 1. 项目目标（Goal）

## 2. 背景与范围（Context & Scope）

- In scope:
- Out of scope:

## 3. 关键约束（Constraints）

- 技术/环境：
- 性能/可靠性：
- 安全/合规：
- 不可改动（Do-not-touch）：

## 4. 已确认的关键决策（Decisions）

- D1: 写清选择及原因、否决替代方案的原因。
- D2:

## 5. 当前代码与结构（Code Map）

- 目录/模块概览：
- 关键文件：
  - `path`: 作用 / 关键函数 / 注意点

## 6. 当前进度（Status）

- 已完成：
- 进行中：
- 未开始：

## 7. 已知问题与风险（Known Issues & Risks）

## 8. 待办清单（Next Actions / TODO）

- [P0] 用动词开头，尽量限定为 1–3 小时可完成。
- [P1]

## 9. 需要我在新会话首先回答的三个问题（Top Questions）

1.
2.
3.

本模板保存在 `docs/agents/HANDOFF_TEMPLATE_CN.md`。`CONTEXT_HANDOFF_SUMMARY.md` 是按需产出的交接记录名称；当前仓库没有该文件，不应把它列为既有必读文件。
