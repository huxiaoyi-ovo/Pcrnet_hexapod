# Final Paper Outputs v3

Generated files:

- fig3_main_performance.pdf
- fig3_main_performance.png
- fig4_mechanism.pdf
- fig4_mechanism.png
- fig6_trajectories_stage4.pdf
- fig6_trajectories_stage4.png
- fig6_trajectories_stage4_sources.csv
- fig6_trajectory_episode_candidates.csv
- table1_main_performance_stage4.csv
- table1_main_performance_stage4.md
- table1_main_performance_stage4.tex
- table2_mechanism_ablation.csv
- table2_mechanism_ablation.md
- table2_mechanism_ablation.tex
- table3_mono_ppo_stage_probe.csv
- table3_mono_ppo_stage_probe.md
- table3_mono_ppo_stage_probe.tex
- tableA1_ppo_hyperparams.tex
- tableA2_reward_terms.tex
- tableA3_domain_randomization.tex
- tableA4_network_structure.tex
- tableA5_delta_y_full.tex

Row counts:

- Table I audit rows: 15
- Table II rows: 4
- Table III rows: 3
- Table A5 rows: 18

Source files and paths:

- internal_all_csv: `agents/eval_data_seed23/pcr_main_table/pcr_main_table_all_metrics_with_rollout_20260529.csv` (mtime=2026-05-29 15:33:18)
- internal_aggregate_csv: `agents/eval_data_seed23/pcr_main_table/pcr_main_table_3seed_aggregate_20260529_manual_seed1.csv` (mtime=2026-05-29 15:31:19)
- risk_all_csv: `agents/eval_data_risk_only/pcr_main_table/pcr_main_table_all_metrics_20260610_170313.csv` (mtime=2026-06-10 17:03:18)
- risk_aggregate_csv: `agents/eval_data_risk_only/pcr_main_table/pcr_main_table_aggregate_20260610_170313.csv` (mtime=2026-06-10 17:03:22)
- rule_all_csv: `agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_all_metrics_20260601_190358.csv` (mtime=2026-06-01 19:04:02)
- rule_aggregate_csv: `agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_aggregate_20260601_190358.csv` (mtime=2026-06-01 19:04:06)
- learnedw_diag_all_csv: `agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_all_metrics_20260602_141845.csv` (mtime=2026-06-02 14:18:46)
- learnedw_diag_aggregate_csv: `agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_aggregate_20260602_141845.csv` (mtime=2026-06-02 14:18:48)
- mono_stage2_path: `agents/eval_data_mono_targetview_stage_probe/stage2_s035` (mtime=2026-06-04 17:27:22)
- mono_stage3_path: `agents/eval_data_mono_targetview_stage_probe/stage3_s035` (mtime=2026-06-04 17:28:17)
- mono_stage4_all_csv: `agents/eval_data_mono_targetview/pcr_main_table/pcr_main_table_all_metrics_20260604_160305.csv` (mtime=2026-06-04 16:03:05)
- learnedw_mechanism_dir: `agents/eval_data/s_0.6/moe_teacher_s_pcr_line_avoid_basic_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_seed1_20260526_204857` (mtime=2026-05-27 14:30:15)
- fig6_timeseries_root: `None` (mtime=missing)

Risk-only source mode:

- requested: `trained`
- resolved: `trained`

Validation checklist:

- Table I: expected 15 rows = 3 speeds x 5 methods; Mono-PPO is intentionally excluded.
- Table I: Risk-only 0.60 success should be about 0.008 +/- 0.008.
- Table II: speed is fixed by --mechanism_speed, default 0.60.
- Table II: Risk-only note must say trained from scratch and no learned-w channel.
- Table II: Params should separate Risk-only and Learned-w.
- Table A5: contains Delta y_w / Delta y_r / Delta y_total over All, C_unsafe, C_avoid.
- Fig.4: x-axis label must be Conflict intensity and Delta y must equal y_eff - y_raw.
- Fig.6: if requested, 0.60 m/s timeseries must contain robot_x/y, target_x/y, obstacles_json, episode_termination_reason, and trajectory_frame=world_xy_train_play; roots may be inferred from main-table CSV sources; default selection uses one shared obstacle layout, requires learned-w success, and prefers 2-3 baseline collisions plus at least one lost/timeout trajectory when available.
- Table III: expected stages 2, 3, and 4; use --allow_incomplete_mono_stage_probe only for local dry-runs.
