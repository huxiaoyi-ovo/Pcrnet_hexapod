# Avoid long-train preflight — 2026-09-11

## Findings

- Planned run: `s_avoid_clutter`, GPU 1, 128 environments, 24 high-level steps and 1000 PPO iterations. The source is the map-raster-fixed snapshot; `model_200.pt` is loaded with `--finetune_from`, so the optimizer is fresh and this is not a resume.
- The run keeps the reviewed entropy `.04`, PPO settings, the other reward and observation definitions, scene generator, safety processing and low-level checkpoint. The scene raster bound fix is the approved map change. The completed Sanity's online result is not a performance gate for this authorized run.
- Periodic checkpoints remain at the existing intervals. The trainer additionally saves the last loop iteration, so a fresh 1000-iteration run produces `model_999.pt` even though 999 is not periodic.
- Run metadata records `num_iterations`, `finetune_from`, and `scene_raster_bounds_version=exclusive_ceil_upper_v1`.
- Source review verifies that rollout stores separate actor/critic maps and the raw 1-D action, and PPO reuses those stored tensors with the validity mask. The clutter request is `[raw lateral action, state[:, 13], 0]`; failure overrides success and terminal reward overrides dense terms, Stage-1 preference reads the pre-step stage, and timeout bootstraps once from the pre-reset observation before the GAE done boundary.
- Source review verifies that curriculum promotion uses only the current stage's complete 200 Decision-history entries with the `.85` success and `.10` failure thresholds. Executed CPU checks were `test_avoid_clutter_training.py`, `test_clutter_env_contract.py`, `test_revision_observation_contract.py`, `test_revision_policy_reuse.py`, and `test_affordance_axis_geometry.py`; all passed. These checks do not establish learned task behavior or real-map correctness.

## Patterns

- A fixed-interval checkpoint schedule can omit a requested final iteration when the final index is not divisible by its interval. The final-iteration condition closes that evidence gap without changing rollout, PPO, reward or reset behavior.
- A weight-only warm start and a resume have different optimizer semantics. The launcher and metadata must preserve that distinction.

## AGENTS Update Proposal

- No update. This is a bounded run-specific checkpoint and provenance correction, not a new project-wide rule.

## TODO Suggestion

- After source review and isolated server synchronization, launch only the approved fresh-optimizer long run and verify that its initial metadata, `model_999.pt`, and controller state match this preflight.
