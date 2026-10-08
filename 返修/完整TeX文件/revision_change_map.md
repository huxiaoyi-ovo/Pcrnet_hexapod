# Revision change map (draft)

This map records the current revised manuscript against protected original `返修/完整TeX文件/final.tex`. The current compiled manuscript is eight pages. Substantive additions/rewrites are blue in the highlighted manuscript. Routine naming, method-label, and formatting changes are not colored individually. No manuscript facts are inferred from missing logs.

## Main scientific and language changes

- **Abstract and Introduction:** lead with person/robot passability mismatch; define the conflict through closed-loop target-retention/clearance consequences. Removed anthropomorphic “risk decides/informs for itself” prose and overbroad wording about waypoint/local navigation. Retained command-space Follow/Avoid framing.
- **Task and metrics:** define desired follow distance as 1.5 m; distinguish event success, continuous follow-error/row-progress metrics, and binary L1/L2/L3 diagnostics and avoid calling row-progress a binary full-row success rate.
- **Directional risk:** identify a robot-fixed forward local grid with camera-mount origin; distinguish the dimensionless occupancy-derived free-space score from metric cone distance measured from local-map origin. Define `rho_F` and `rho_A`; state that nonzero translation uses a directional cone while the historical simulation's near-zero translation falls back to minimum occupied distance. Neither query certifies swept yaw/body clearance. Thresholds are fixed design heuristics, not optimal values.
- **Memory and side visibility:** distinguish current map evidence from the gate's scalar forward-distance-decayed risk memory and state that the memory's isolated causal contribution was not evaluated. Avoid proposes lateral motion from visible forward geometry; no complete side/rear coverage is claimed. Historical real-trial temporal map fusion is unknown; the manuscript does not assert either temporal fusion or a fresh-only map for those trials.
- **Training:** identify the shared Isaac Gym hexapod asset and RSL-RL/PPO stage roles; report actor/critic architecture, observation meanings, outputs, allocated interaction budgets, verified Avoid reward categories, and terminal events. Gate rewards exclude switch penalty, and the BCE term is a separate auxiliary loss; row-release signals are auxiliary labels only, not actor input. Avoid reward coefficients are not tied to the checkpoint because its full reward snapshot/source hash is absent.
- **Comparisons and interpretation:** Risk-only retains learned base `alpha` plus analytic `Delta alpha_r`, retrained without `w`; comparisons share base sensor inputs. Remove unverified 3.28% parameter-difference gloss and state actor--critic counts (including training-only critics). Limit Additive-Fusion/fixed authority and Mono conclusions to tested conditions and the single Mono training seed.
- **Sensitivity:** retain the separate seed-101, five-setting, three-speed, 128-episode-per-setting/speed diagnostic; state its mixed-difficulty pooled success/collision ranges (1.56/1.30 pp). It is not Table I's held-out Stage 4 statistic and does not establish equivalence.
- **Hardware:** preserve 40 Adaptive and 20 Rule-Override trial counts by scene; replace “matched trials” with staggered-row trial wording. State only the confirmed stereo-depth map versus simulation actor-rasterized map distinction and separate scalar gate memory. Original launch/log configuration and Rule-Override perception parity remain unverified.
- **Network specifications (Sec. IV-F):** three labeled paragraphs describe map encoding, scalar encoding and fusion, and output heads and critics. All network dimensions and parameter counts from the former architecture table are retained. Avoid and gate have separate encoders; four gate heads parameterize two Beta distributions. Counts include training-only critics. Only the evaluated gate is documented as having a zero-valued difficulty slot; no claim is made about the Avoid slot. Parameterization source is not bound into checkpoint metadata.
- **Tables I--V:** main performance, inference ablation, Mono, DWA, and hardware, respectively. Removing the architecture table changes numbering only; numerical results and best-mean markers are unchanged.
- **Related Work/references:** reorder existing bibliography by first citation without changing keys or deleting citations. Add bounded motivation for reward shaping relative to explicit constraint budgets. New citations and substantive introductory text remain blue.
- **Limitations/conclusion:** discuss a target stopping before a barrier as a conditional limitation (not a reported trial), no embodiment ablation, single Mono training seed, and structured scene scope; avoid universal or safety-guarantee claims.

## Reviewer response synchronization

- `author_response.tex` addresses AE, R1, R2, and R10 comments and locates changes by revised section and page. Network specifications are in Sec. IV-F. Fig. 5 presents Mono capability probes; Table III gives the formal held-out Mono comparison, and Table I gives the main arbitration results.
- No hover annotations, deletion-strike convention, or journal-mandated blue color are asserted. The response explains the adopted blue convention and the RA-L combined response/highlighted manuscript format.
- Draft remains pending author checks: full Avoid reward snapshot if available, original 60-trial map/perception launcher records, Fig. 2 artwork, and supplementary-video review.

## Evidence limits and removed claims

- Avoid checkpoint: `/home/dell/RL_hexapod_gym/agents/avoid_best.pt`, SHA256 `fd23cebc9a0f1d7a280dd7561accdebccf4f6de0f41787a2214c6e9439adf3df`. Checkpoint tensor shapes were inspected on CPU. Apr 3 run metadata exists at `/home/dell/RL_hexapod_gym/outputs/planner/avoid_teacher_20260403_100941/run_meta.json`; nearby source commit `49235d2` does not bind the checkpoint to a complete reward snapshot. Best iteration 949 does not establish the allocated training budget. Exact historical reward coefficients and rationale for keeping a 3-D head are not claimed.
- Historical Avoid architecture evidence: 4-channel map conv weights `(32,4,3,3),(64,32,3,3),(128,64,3,3)`, projection `(128,2048)`, state `(64,14)` then 64/64, goal `(32,2)` then 32, fusion `(256,225)` then 256, mean/std action heads `256->64->3`, mirrored critic encoders and value `256->64->1`.
- Gate architecture evidence: map same; state input 13, goal/conflict input 18; four shape heads `y_alpha/y_beta/w_alpha/w_beta`, each `256->64->1`, plus Softplus in source. Distinct actor and critic encoders are present. Source code defines ELU and pooling, but checkpoint metadata does not carry source commit.
- Map evidence: origin is `camera_mount`; free-space score is not metric distance. Prior source commit `6102102` documents occupied dilation, free-mask generation, local 3x3 averaging, cellwise maximum with the free mask, and masking. Historical real 60-trial launch/logs were not located; current defaults are not treated as historical evidence.
- Risk sensitivity is a separate seed-101 mixed-difficulty diagnostic. It is not treated as formal Table I Stage 4 evidence or statistical equivalence.
- Removed claims: unsupported short-temporal-history implementation for the historical 60 trials; unsupported assertion of matched perception for both hardware methods; “risk chooses/decides” anthropomorphism; unsupported broad local-navigation limitation; assertion that gate switch penalty contributes when its coefficient is zero; universal necessity, safety, or convergence claims.

## Official submission-format reference

IEEE RA-L author information: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/. The response is followed by the highlighted manuscript in a combined PDF per the author instructions. No hover annotations, mandatory blue hue, or deletion strike-through is stated in the instruction used for this revision.

## Final compression and interpretation (2026-09-24)

- Preserved all figure sizes, the new network table, the Mono interpretation, metric values, font sizes, and page geometry. Removed repeated framework explanations, repeated signed-gate interpretation, and duplicated sensing descriptions.
- Restored compact `{\pm}` spacing in Table II; used 1.18 row spacing and 1.95 pt column spacing without resizing its text. Adaptive means remain bold; best-mean daggers do not denote statistical significance.
- Expanded Mono analysis in Results, Discussion, and Conclusion: the extended run is consistent with constituent capabilities emerging before reliable complete-task coordination. The proposed structure concentrates learning on authority assignment over fixed behaviors. No new experiment, plot, or optimizer-mechanism claim was added.
- The response contains AE plus R1.1–R1.16, R2.1–R2.28, and R10.1–R10.3, matching the issue audit and current page locations.
- New and revised references remain highlighted; mechanical method renaming is not individually colored. No hover comments were added.

## Mono checkpoint analysis update (2026-09-24)

- Verified the 4,000-iteration Stage 4 result at 0.50 m/s against the per-seed CSV and remote `metrics.json` records: seeds 101, 102, and 103 each have 121/128 strict successes (94.53125%), yielding 94.5% mean and 0.0 sample SD. The same checkpoint file is used across the three speed conditions for each reported iteration. This records the evaluation design only; it does not establish pre-registration or rule out post hoc checkpoint selection.
- Replaced the vague Level-4 summary with the observed cross-speed transfer and non-monotonic held-out interpretation. The cause of regression is not isolated; no training-seed variability or full Mono potential is claimed.
- Updated the Discussion to connect Additive-Fusion, Fixed-Authority, the inference-only signed Follow-authority correction ablation, and Mono through a task-relevant inductive-bias interpretation. The comparisons are not presented as causal isolation.
- Added a blue Table IV note for Adaptive's two-stage interaction budget, excluded shared low-level training, the single Mono training seed, sample-SD evaluation across three seeds, and shared checkpoint per speed row.

## Network presentation update (2026-09-24)

- Replaced the architecture table with three labeled paragraphs in Sec. IV-F, retaining the reported inputs, layer dimensions, activations, fusion dimensions, executed outputs, and actor--critic counts.
- Earlier dated entries describe the preceding table-based layout. Current tables are I (main), II (ablation), III (Mono), IV (DWA), and V (hardware).

## Reviewer-response wording audit (2026-09-24)

- Clarified the geometric distinction between training and evaluation: mirrored alternating training passages versus L-L-R-R-L evaluation passages, with explicit adjacent-row distances and a geometry fixed across evaluation seeds. Retained the author-confirmed 15 obstacles, shared formal layout, and 0.50 m/s training cap.
- Removed the 0.75 m minimum-passage value because it was not established for the current 15-obstacle layout. Source: legged_gym/envs/hex_v4/layouts/revision_heldout_mixed_v1.json. No replacement gap-width claim was inferred.
- Restored L3 (final-row clearance), retaining the Metrics distinction from continuous row progress. Clarified absolute bearing thresholds, the signed-modulation deadband, and the implementation-only rationale for retaining unused Avoid outputs.
- Stated that no dedicated dead-end recovery or global replanning mechanism is implemented; synchronized the corresponding replies and YOLO's target-detection role.
- Substantive clarifications remain blue; restored terminology and minor wording adjustments are black. Existing figures, numerical results, and the protected original are unchanged.
## Author clarification of final evaluation (2026-09-25)

- The author clarified that the final experiments were run on another machine and that all formal simulation comparisons use held-out Stage 4 rather than the training layouts. The setup, captions, Results, Mono comparison, and response now follow that final protocol.
- Removed the prior evaluation-attribution blocker from manuscript comments and the response; historical archive findings remain in reviewer_issue_audit.md with their applicability superseded. This is an author-provided protocol clarification, not a claim that the unseen final-run logs were independently verified.
- Retained 15 capsules, the non-alternating test arrangement, explicit row spacings, the 0.50 m/s training cap, and the distinction between formal comparisons and diagnostic probes. The single-layout limitation is explicit. Numerical results, figures, and the protected original are unchanged.
