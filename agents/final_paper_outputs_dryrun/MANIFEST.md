# Final Paper Outputs

Generated files:

- table1_main_performance_stage4.csv
- table1_main_performance_stage4.md
- fig4_speed_curves_stage4.png
- fig4_speed_curves_stage4.pdf
- table2_mechanism_ablation.csv
- table2_mechanism_ablation.md
- table3_mono_ppo_stage_probe.csv
- table3_mono_ppo_stage_probe.md

Row counts:

- Table I: 15 rows
- Table II: 12 rows
- Table III: 1 rows

Source files and paths:

- internal_all_csv: `agents/eval_data_seed23/pcr_main_table/pcr_main_table_all_metrics_with_rollout_20260529.csv` (mtime=2026-05-29 15:35:34)
- internal_aggregate_csv: `agents/eval_data_seed23/pcr_main_table/pcr_main_table_3seed_aggregate_20260529_manual_seed1.csv` (mtime=2026-05-29 15:11:17)
- risk_all_csv: `agents/eval_data_risk_only/pcr_main_table/pcr_main_table_all_metrics_20260601_223036.csv` (mtime=2026-06-02 12:23:12)
- risk_aggregate_csv: `agents/eval_data_risk_only/pcr_main_table/pcr_main_table_aggregate_20260601_223036.csv` (mtime=2026-06-02 12:23:12)
- rule_all_csv: `agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_all_metrics_20260601_190358.csv` (mtime=2026-06-01 20:05:26)
- rule_aggregate_csv: `agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_aggregate_20260601_190358.csv` (mtime=2026-06-01 20:05:26)
- learnedw_diag_all_csv: `agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_all_metrics_20260602_141845.csv` (mtime=2026-06-02 15:52:20)
- learnedw_diag_aggregate_csv: `agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_aggregate_20260602_141845.csv` (mtime=2026-06-02 15:52:20)
- mono_stage2_path: `agents/eval_data_mono_targetview_stage_probe/stage2_s035` (mtime=2026-06-04 17:54:31)
- mono_stage3_path: `agents/eval_data_mono_targetview_stage_probe/stage3_s035` (mtime=2026-06-04 17:54:31)
- mono_stage4_all_csv: `agents/eval_data_mono_targetview/pcr_main_table/pcr_main_table_all_metrics_20260604_160305.csv` (mtime=2026-06-04 16:26:42)
- learnedw_mechanism_dir: `agents/eval_data/s_0.6/moe_teacher_s_pcr_line_avoid_basic_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_seed1_20260526_204857` (mtime=2026-05-27 14:55:20)

Validation checklist:

- Fig.4: legend must contain Y-only / Geom-w / Risk-only / Rule-Override / Learned-w.
- Table I: expected 15 rows = 3 speeds x 5 methods; Mono-PPO is intentionally excluded.
- Table II: Risk-only Delta y_w@C_avoid should be 0.000; Learned-w should be non-zero.
- Table III: expected stages 2, 3, and 4; one-row stage4 output is not a valid paper table.
- Source timestamps: verify all source paths are the intended final eval outputs.
