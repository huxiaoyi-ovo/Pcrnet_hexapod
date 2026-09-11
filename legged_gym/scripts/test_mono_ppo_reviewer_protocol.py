"""Static regression checks for the Reviewer-proof Mono-PPO protocol.

This file intentionally imports neither Torch nor Isaac Gym.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = (ROOT / "legged_gym/scripts/train_highlevel.py").read_text(encoding="utf-8")
PLANNER = (ROOT / "rsl_rl/algorithms/high_level_planner.py").read_text(encoding="utf-8")
EVAL = (ROOT / "legged_gym/scripts/eval_highlevel.py").read_text(encoding="utf-8")
RUNNER = (ROOT / "legged_gym/scripts/run_pcr_main_table_eval.py").read_text(encoding="utf-8")


def require(text: str, label: str) -> None:
    if text not in TRAIN:
        raise AssertionError(f"missing protocol invariant: {label}")


def main() -> None:
    for text, label in (
        ('task != "s_pcr_new"', "canonical training task"),
        ('"num_iterations", 3000', "3BG default"),
        ('"mini_batch_size", 3072', "four minibatches"),
        ('"lr", 3e-4', "Mono learning rate"),
        ('"clip_range", 0.20', "PPO clip range"),
        ('"value_loss_coef", 1.0', "value coefficient"),
        ('"max_grad_norm", 1.0', "gradient limit"),
        ('"lr_schedule", "adaptive"', "adaptive KL schedule"),
        ('"desired_kl", 0.01', "adaptive KL target"),
        ('"use_clipped_value_loss", True', "clipped value loss"),
        ('completed_iteration in (1000, 2000, 3000)', "unambiguous reviewer checkpoints"),
        ('reviewer_mono and completed_iteration % 100 == 0', "strict completed-iteration cadence"),
        ('gate_stage_interaction_reference', "gate-stage reference alias"),
        ('pcr_total_interaction_reference', "PCR-total reference alias"),
        ('full_training', "fixed full-training alias"),
        ('"pcr_avoid_pretrain_interactions"', "Avoid pretraining cost provenance"),
        ('pcr_total_completed_iteration != 2000', "exact PCR-total checkpoint guard"),
        ('"elapsed_wall_seconds"', "checkpoint wall-time evidence"),
        ('reviewer_mono_formal and resume_path', "formal resume rejection"),
        ('必须从 completed iteration 0 绝对连续训练到 3000', "absolute 3000 hard-stop guard"),
        ('total_iterations = 3000', "absolute reviewer total iterations"),
        ('wall_time_base_seconds + (time.time() - training_wall_start)', "linear wall-time accumulation"),
        ('Reviewer-proof Mono-PPO frozen profile violation:', "frozen profile fail-closed error"),
        ('"seed": (getattr(args, "seed", None), (1, 2, 3))', "formal seed set"),
        ('"pcr_avoid_pretrain_interactions": (getattr(args, "pcr_avoid_pretrain_interactions", None), 12_288_000)', "frozen Avoid cost"),
        ('math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)', "exact frozen float comparison"),
        ('lr_schedule must be adaptive', "frozen adaptive schedule"),
        ('use_clipped_value_loss must be true', "frozen clipped value loss"),
        ('meta["source_git_commit"] == "unknown" or meta["source_git_dirty"] is not False', "formal clean-source guard"),
        ('需要已固定且干净的 Git 源码树', "formal provenance fail-closed error"),
        ('(not reviewer_mono_formal) and online_selection_metric > current_best_tuple', "formal online-best checkpoint rejection"),
        ('revision_contract 的正式 PCR/Mono 禁止 --gate_use_difficulty', "revision difficulty rejection"),
        ('"gate_smooth",', "reward audit gate smooth key"),
        ('"pcr_gate_aux",', "reward audit gate auxiliary key"),
        ('"pcr_yaw_suppress",', "reward audit yaw key"),
        ('legacy_follow_removed={bool(self.is_pcr_line_task and (not bool(getattr(self, \'mono_ppo_direct_cmd\', False)) or bool(getattr(self.args, \'revision_contract\', False))))}', "revision legacy-follow audit flag"),
        ('ReviewerMono/CumulativeEnvironmentInteractions', "cumulative environment interactions log"),
        ('ReviewerMono/CumulativeOptimizerUpdates', "cumulative optimizer updates log"),
        ('ReviewerMono/CumulativeWallTimeHours', "cumulative GPU wall time log"),
        ('ReviewerMono/LatentGaussianActionStdMean', "Gaussian std log"),
        ('ReviewerMono/NormalizedActionSaturationRate', "action saturation log"),
        ('ReviewerMono/PostprocessorRawToClampedRate', "postprocessor clamp log"),
        ('[Mono-PPO SharedReward]', "shared reward console label"),
        ('disabled: legacy approach/follow_outside/target-view, gate aux, gate smooth, yaw suppress', "shared reward exclusions console label"),
        ('"reward_contract_version"', "reward provenance"),
        ('"source_git_commit"', "source provenance"),
        ('"config_sha256"', "configuration hash"),
        ('"optimizer_updates_completed"', "completed optimizer updates"),
        ('"pcr_common_task_reward_v1"', "common reward contract"),
    ):
        require(text, label)
    if 'pcr_yaw_suppress_scale = 0.0 if bool(getattr(self.args, "revision_contract", False))' not in TRAIN:
        raise AssertionError("revision contract must disable yaw suppression")
    if 'env.disable_pcr_gate_aux = bool(is_mono_ppo or getattr(args, "revision_contract", False))' not in TRAIN:
        raise AssertionError("revision contract must disable PCR gate auxiliary reward")
    if "reward_dict['gate_smooth'] = torch.zeros_like(gate_smooth)" not in TRAIN:
        raise AssertionError("revision contract must remove gate-only smooth reward")
    if '"gate_smooth", "pcr_gate_aux"' not in TRAIN:
        raise AssertionError("reward metadata must exclude gate smooth reward")
    for key in ('"lr_schedule",', '"desired_kl",', '"use_clipped_value_loss",'):
        if key not in TRAIN:
            raise AssertionError(f"metadata must track {key}")
    if 'reviewer_mono_actor_difficulty_zero = bool(is_mono_ppo and getattr(args, "revision_contract", False))' not in TRAIN:
        raise AssertionError("training Mono revision actor difficulty must be zero")
    if '"actor_difficulty_semantics"' not in TRAIN or '"critic_difficulty_semantics"' not in TRAIN:
        raise AssertionError("difficulty semantics must be recorded in metadata")
    if '"zero" if is_mono_ppo' not in TRAIN:
        raise AssertionError("reviewer Mono critic difficulty must be zero")
    if 'critic_difficulty = torch.zeros_like(critic_difficulty)' not in TRAIN:
        raise AssertionError("rollout critic difficulty must be zero for reviewer Mono")
    if 'bootstrap_critic_difficulty = torch.zeros_like(bootstrap_critic_difficulty)' not in TRAIN:
        raise AssertionError("timeout bootstrap critic difficulty must be zero for reviewer Mono")
    if 'critic_difficulty_next = torch.zeros_like(critic_difficulty_next)' not in TRAIN:
        raise AssertionError("next-value critic difficulty must be zero for reviewer Mono")
    if 'self.mono_reviewer_actor_difficulty_zero' not in EVAL:
        raise AssertionError("evaluation must replay zero Mono actor difficulty")
    if 'not bool(getattr(args, "mono_ppo", False)) and not bool(getattr(args, "revision_contract", False))' in TRAIN:
        raise AssertionError("unexpected legacy target-reward condition")
    if '"cmd_mean": out.cmd_mean, "cmd_std": out.cmd_std' not in PLANNER:
        raise AssertionError("CmdVelExpert must expose Gaussian parameters for exact KL")
    if 'log_scale_jacobian = torch.log(self.cmd_scale.abs().clamp_min(1e-6)).view(1, -1)' not in PLANNER:
        raise AssertionError("tanh Gaussian log-prob must include command-scale Jacobian")
    if 'cmd_log_prob = (log_prob_raw - log_det_jacobian - log_scale_jacobian).sum(dim=-1)' not in PLANNER:
        raise AssertionError("command-scale Jacobian must affect absolute action log-prob")
    if 'cmd_entropy = -(cmd_entropy_log_prob_raw - cmd_entropy_log_det - log_scale_jacobian).sum(dim=-1)' not in PLANNER:
        raise AssertionError("command-scale Jacobian must affect squashed entropy")
    for text, label in (
        ('--reviewer_mono_validation', "reviewer validation preset"),
        ('methods != ["mono_ppo"]', "validation method guard"),
        ('expected_seeds = [101, 102]', "validation seed guard"),
        ('expected_speeds = [0.35, 0.50]', "validation speed guard"),
        ('int(args.num_envs) != 64 or int(args.episodes) != 128', "validation resource and episode guard"),
        ('expected_difficulty_levels = [0.0, 0.25, 0.5, 0.75, 1.0]', "validation difficulty guard"),
        ('int(args.timeseries_episodes) != 64 or int(args.timeseries_stride) != 1', "validation timeseries guard"),
        ('禁止 --eval_layout/heldout', "validation layout guard"),
        ('not for early stop, checkpoint selection, final test', "offline-only validation notice"),
    ):
        if text not in RUNNER:
            raise AssertionError(f"missing reviewer validation invariant: {label}")
    print("PASS: Reviewer-proof Mono-PPO static protocol checks")


if __name__ == "__main__":
    main()
