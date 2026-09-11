"""Pure-Torch checkpoint and frozen Avoid reuse rules for the revised PCR path."""

import torch

from legged_gym.pcr_observation import AVOID_ACTOR_MASK_INDICES, build_avoid_actor_state, side_preference


AVOID_1D_REVISION = "avoid_1d_canonical_v1"
PCR_CANONICAL_REVISION = "pcr_canonical_v1"


def validate_frozen_avoid_revision(primary_meta, avoid_meta):
    """Fail closed on a legacy/revised Gate-or-Mono and Avoid mix."""
    primary = (primary_meta or {}).get("revision_contract")
    avoid = (avoid_meta or {}).get("revision_contract")
    legacy_primary = primary is None
    legacy_avoid = avoid is None
    if legacy_primary and legacy_avoid:
        return
    if primary == PCR_CANONICAL_REVISION and avoid == AVOID_1D_REVISION:
        return
    raise ValueError("primary PCR revision and frozen Avoid revision must be both legacy or both canonical")


def _state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dictionary")
    return ckpt.get("model_state_dict", ckpt)


def _meta(ckpt):
    return ckpt.get("experiment_meta", {}) if isinstance(ckpt, dict) else {}


def checkpoint_cmd_policy_kwargs(ckpt, cmd_scale):
    """Infer only the frozen command head contract; reject ambiguous 1-D weights."""
    state = _state_dict(ckpt)
    weight = state.get("cmd_mean_head.2.weight")
    if weight is None or weight.ndim != 2:
        raise ValueError("checkpoint lacks cmd_mean_head.2.weight")
    action_dim = int(weight.shape[0])
    meta = _meta(ckpt)
    if action_dim == 3:
        if meta.get("revision_contract") == PCR_CANONICAL_REVISION:
            return {"action_dim": 3, "cmd_scale": tuple(cmd_scale), "actor_state_mask_indices": (0, 1)}
        return {"action_dim": 3, "cmd_scale": tuple(cmd_scale)}
    if action_dim != 1:
        raise ValueError(f"unsupported command output dimension: {action_dim}")
    if (
        meta.get("revision_contract") != AVOID_1D_REVISION
        or int(meta.get("actor_output_dim", -1)) != 1
        or int(meta.get("policy_goal_dim", meta.get("goal_dim", -1))) != 2
        or list(meta.get("actor_state_mask_indices", ())) != list(AVOID_ACTOR_MASK_INDICES)
    ):
        raise ValueError("1-D Avoid checkpoint is missing the frozen avoid_1d_canonical_v1 contract")
    state_weight = state.get("state_encoder.mlp.0.weight")
    goal_weight = state.get("goal_encoder.mlp.0.weight")
    if state_weight is None or int(state_weight.shape[1]) != 14:
        raise ValueError("1-D Avoid checkpoint must use state_dim=14")
    if goal_weight is None or int(goal_weight.shape[1]) != 2:
        raise ValueError("1-D Avoid checkpoint must use goal_dim=2")
    return {
        "action_dim": 1,
        "cmd_scale": (float(cmd_scale[0]),),
        "actor_state_mask_indices": tuple(AVOID_ACTOR_MASK_INDICES),
    }


def get_avoid_command(model, affordance_map, raw_state, follow_cmd, follow_goal, difficulty, *, deterministic=True, legacy_state=None, legacy_goal=None, target_valid=None):
    """Return a 3-D PCR command while retaining a legacy model's old inputs exactly."""
    if int(getattr(model, "action_dim", 3)) == 1:
        state = build_avoid_actor_state(raw_state, follow_cmd[:, 1])
        goal = side_preference(follow_goal, valid=target_valid)
        lateral, info = model.get_action(affordance_map, state, goal, difficulty, deterministic=deterministic)
        return torch.cat([lateral, torch.zeros_like(lateral), torch.zeros_like(lateral)], dim=1), info
    state = raw_state if legacy_state is None else legacy_state
    goal = follow_goal if legacy_goal is None else legacy_goal
    return model.get_action(affordance_map, state, goal, difficulty, deterministic=deterministic)
