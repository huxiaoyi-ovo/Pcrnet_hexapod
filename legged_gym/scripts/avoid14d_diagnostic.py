"""Optional, counterfactual-only recorder for the frozen Avoid 14th state slot.

The collector never supplies a probe action to the environment.  It records the
normal rollout's pre-step state and calls the frozen Avoid expert separately for
four copies that differ only at state index 13.
"""

import json
import os

import numpy as np
import torch


PROBE_VALUES = np.asarray((0.0, 0.20, 0.35, 0.50), dtype=np.float32)
EPSILON_UX = 0.02
CATEGORY_NAMES = (
    "all",
    "straight",
    "lateral",
    "turning",
    "near_follow_proposal",
    "row_transition",
)


def validate_probe_args(args, avoid_state_dim=None):
    """Reject probe configurations outside the frozen Avoid comparison."""
    if str(getattr(args, "task", "")) != "s_pcr_line_avoid_basic":
        raise ValueError("--avoid14d_probe_dir requires --task s_pcr_line_avoid_basic")
    if str(getattr(args, "skill", "")) != "moe" or bool(getattr(args, "mono_ppo", False)):
        raise ValueError("--avoid14d_probe_dir requires frozen MoE, not Mono-PPO")
    if bool(getattr(args, "generalize", False)) or str(getattr(args, "eval_layout", "") or "").strip():
        raise ValueError("--avoid14d_probe_dir only supports the standard layout")
    speed = getattr(args, "pcr_line_target_speed", None)
    if speed is not None and not any(abs(float(speed) - value) < 1e-7 for value in (0.35, 0.50)):
        raise ValueError("--avoid14d_probe_dir only supports --pcr_line_target_speed .35 or .50")
    if getattr(args, "pcr_line_target_speed_scale", None) is not None:
        raise ValueError("--avoid14d_probe_dir does not support --pcr_line_target_speed_scale")
    if int(getattr(args, "avoid14d_probe_stride", 4)) < 1:
        raise ValueError("--avoid14d_probe_stride must be >= 1")
    max_samples = int(getattr(args, "avoid14d_probe_max_samples", 8192))
    if max_samples < 1 or max_samples > 8192:
        raise ValueError("--avoid14d_probe_max_samples must be in [1, 8192]")
    if avoid_state_dim is not None and int(avoid_state_dim) != 14:
        raise ValueError(f"--avoid14d_probe_dir requires Avoid state_dim=14, got {avoid_state_dim}")


def build_candidate_states(expert_state, probe_values=PROBE_VALUES):
    """Return [env, probe, 14] copies differing only in the final state slot."""
    if expert_state.ndim != 2 or int(expert_state.shape[1]) != 14:
        raise ValueError(f"Avoid probe expects [N,14] expert_state, got {tuple(expert_state.shape)}")
    values = torch.as_tensor(probe_values, device=expert_state.device, dtype=expert_state.dtype)
    candidates = expert_state.detach().unsqueeze(1).expand(-1, int(values.numel()), -1).clone()
    candidates[:, :, 13] = values.view(1, -1)
    return candidates


def candidate_actions(model, aff_map, expert_state, goal, difficulty, probe_values=PROBE_VALUES):
    """Evaluate frozen candidates without changing any live action or input tensor."""
    candidates = build_candidate_states(expert_state, probe_values)
    env_count, probe_count = candidates.shape[:2]
    with torch.no_grad():
        actions, _ = model.get_action(
            aff_map.detach().repeat_interleave(probe_count, dim=0),
            candidates.reshape(env_count * probe_count, -1),
            goal.detach().repeat_interleave(probe_count, dim=0),
            difficulty.detach().repeat_interleave(probe_count, dim=0),
            deterministic=True,
        )
    return candidates, actions.reshape(env_count, probe_count, -1)


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_candidates(candidate_cmd, category_masks, probe_values=PROBE_VALUES):
    """Summarize Ux deltas; rollout samples are correlated states, not iid trials."""
    cmd = np.asarray(candidate_cmd, dtype=np.float64)
    if cmd.ndim != 3 or cmd.shape[1] != len(probe_values) or cmd.shape[2] < 1:
        raise ValueError("candidate_cmd must have shape [sample, 4, action_dim]")
    base = cmd[:, 0, 0]
    out = {
        "sample_dependence": "rollout states are correlated; no iid inference or p-values are reported",
        "epsilon_ux_mps": EPSILON_UX,
        "categories": {},
    }
    for name in CATEGORY_NAMES:
        mask = np.asarray(category_masks.get(name, np.zeros(cmd.shape[0], dtype=bool)), dtype=bool)
        if mask.shape != (cmd.shape[0],):
            raise ValueError(f"category {name} has incompatible mask shape {mask.shape}")
        category = {"count": int(mask.sum()), "missing": not bool(mask.any()), "probes": {}}
        base_cat = base[mask]
        for probe_idx, probe in enumerate(probe_values):
            candidate = cmd[mask, probe_idx, 0]
            finite_pair = np.isfinite(base_cat) & np.isfinite(candidate)
            base_valid = finite_pair & (np.abs(base_cat) > EPSILON_UX)
            candidate_valid = finite_pair & (np.abs(candidate) > EPSILON_UX)
            both_valid = base_valid & candidate_valid
            raw_sign_flip = finite_pair & (base_cat * candidate < 0.0)
            valid_sign_flip = both_valid & (base_cat * candidate < 0.0)
            baseline_nearzero = finite_pair & (np.abs(base_cat) <= EPSILON_UX)
            becomes_nearzero = base_valid & (np.abs(candidate) <= EPSILON_UX)
            crosses_threshold = finite_pair & ((np.abs(base_cat) <= EPSILON_UX) != (np.abs(candidate) <= EPSILON_UX))
            category["probes"][f"{float(probe):.2f}"] = {
                "abs_delta_ux_mps": _finite_summary(np.abs(candidate - base_cat)),
                "finite_pair_count": int(finite_pair.sum()),
                "baseline_valid_count": int(base_valid.sum()),
                "both_valid_count": int(both_valid.sum()),
                "valid_sign_flip_count": int(valid_sign_flip.sum()),
                "valid_sign_flip_rate": (float(valid_sign_flip.sum() / both_valid.sum()) if both_valid.any() else None),
                "raw_sign_flip_count": int(raw_sign_flip.sum()),
                "baseline_nearzero_count": int(baseline_nearzero.sum()),
                "becomes_nearzero_count": int(becomes_nearzero.sum()),
                "crosses_nearzero_threshold_count": int(crosses_threshold.sum()),
            }
        out["categories"][name] = category
    return out


class Avoid14DProbeRecorder:
    """Small pre-step recorder; row history is reset before the next episode."""

    def __init__(self, out_dir, num_envs, stride=4, max_samples=8192):
        self.out_dir = str(out_dir)
        self.num_envs = int(num_envs)
        self.stride = int(stride)
        self.max_samples = int(max_samples)
        self.global_step = 0
        self.episode_id = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_step = np.zeros(self.num_envs, dtype=np.int64)
        self._previous_row = np.full(self.num_envs, -1, dtype=np.int64)
        self._previous_row_valid = np.zeros(self.num_envs, dtype=bool)
        self._fields = {}

    @property
    def sample_count(self):
        return sum(chunk.shape[0] for chunk in self._fields.get("env_id", ()))

    def update_rows(self, row_index, row_valid):
        rows = np.asarray(row_index, dtype=np.int64).reshape(self.num_envs)
        valid = np.asarray(row_valid, dtype=bool).reshape(self.num_envs)
        transition = valid & self._previous_row_valid & (rows != self._previous_row)
        self._previous_row = rows.copy()
        self._previous_row_valid = valid.copy()
        return transition

    def should_sample(self, row_transition):
        if self.sample_count >= self.max_samples:
            return np.zeros(self.num_envs, dtype=bool)
        regular = (self.global_step % self.stride) == 0
        return np.asarray(row_transition, dtype=bool) | regular

    def record(
        self,
        selected_ids,
        *,
        state13,
        expert_state14,
        aff_map,
        goal,
        difficulty,
        candidate_cmd,
        live_cmd_a,
        base_vel_pre,
        row_index,
        row_valid,
        clearance_follow_proposal,
        row_transition,
    ):
        remain = self.max_samples - self.sample_count
        selected_ids = np.asarray(selected_ids, dtype=np.int64).reshape(-1)[:max(0, remain)]
        if selected_ids.size == 0:
            return
        def grab(value, dtype=None):
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            return np.asarray(array[selected_ids], dtype=dtype)
        velocity = grab(base_vel_pre, np.float32)
        clearance = grab(clearance_follow_proposal, np.float32).reshape(-1)
        cats = {
            "straight": (velocity[:, 1] > 0.05) & (np.abs(velocity[:, 0]) < 0.05) & (np.abs(velocity[:, 2]) < 0.10),
            "lateral": np.abs(velocity[:, 0]) >= 0.05,
            "turning": np.abs(velocity[:, 2]) >= 0.20,
            "near_follow_proposal": clearance <= 0.57,
            "row_transition": grab(row_transition, bool).reshape(-1),
        }
        fields = {
            "env_id": selected_ids.astype(np.int64),
            "episode_id": self.episode_id[selected_ids].copy(),
            "episode_step": self.episode_step[selected_ids].copy(),
            "global_step": np.full(selected_ids.size, self.global_step, dtype=np.int64),
            "state13": grab(state13, np.float32),
            "expert_state14": grab(expert_state14, np.float32),
            "aff_map": grab(aff_map, np.float32),
            "goal": grab(goal, np.float32),
            "difficulty": grab(difficulty, np.float32),
            # Candidate inference is intentionally run only for selected_ids.
            "candidate_cmd": np.asarray(
                candidate_cmd.detach().cpu().numpy() if torch.is_tensor(candidate_cmd) else candidate_cmd,
                dtype=np.float32,
            ),
            "live_cmd_a": grab(live_cmd_a, np.float32),
            "base_vel_pre": velocity,
            "row_index": grab(row_index, np.int64).reshape(-1),
            "row_valid": grab(row_valid, bool).reshape(-1),
            "clearance_follow_proposal_pre": clearance,
            "category_straight": cats["straight"],
            "category_lateral": cats["lateral"],
            "category_turning": cats["turning"],
            "category_near_follow_proposal": cats["near_follow_proposal"],
            "category_row_transition": cats["row_transition"],
        }
        for name, value in fields.items():
            self._fields.setdefault(name, []).append(value)

    def advance_after_step(self, dones):
        done = np.asarray(dones.detach().cpu().numpy() if torch.is_tensor(dones) else dones, dtype=bool).reshape(self.num_envs)
        self.global_step += 1
        self.episode_step += 1
        self.episode_id[done] += 1
        self.episode_step[done] = 0
        self._previous_row_valid[done] = False
        self._previous_row[done] = -1

    def finalize(self, metadata):
        arrays = {name: np.concatenate(parts, axis=0) for name, parts in self._fields.items()}
        count = int(arrays.get("env_id", np.empty(0)).shape[0])
        categories = {"all": np.ones(count, dtype=bool)}
        for name in CATEGORY_NAMES[1:]:
            categories[name] = arrays.get(f"category_{name}", np.zeros(count, dtype=bool))
        candidate_cmd = arrays.get("candidate_cmd", np.empty((0, len(PROBE_VALUES), 3), dtype=np.float32))
        summary = summarize_candidates(candidate_cmd, categories)
        summary.update({
            "sample_count": count,
            "probe_values": [float(v) for v in PROBE_VALUES],
            "category_definitions": {
                "straight": "pre-step vy>0.05, abs(vx)<0.05, abs(wz)<0.10 m/s or rad/s",
                "lateral": "pre-step abs(vx)>=0.05 m/s",
                "turning": "pre-step abs(wz)>=0.20 rad/s",
                "near_follow_proposal": "pre-step follow-proposal directional clearance_F<=0.57 m; not body distance",
                "row_transition": "pre-step current valid row index differs from previous step in same episode",
            },
            "metadata": metadata,
        })
        os.makedirs(self.out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(self.out_dir, "avoid14d_samples.npz"), **arrays)
        with open(os.path.join(self.out_dir, "avoid14d_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
