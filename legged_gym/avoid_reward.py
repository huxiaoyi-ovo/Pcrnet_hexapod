"""Pure terminal-first reward for the standalone 1-D clutter Avoid task."""

import torch


def avoid_clutter_reward(
    progress_delta, clearance, raw_lateral, executed_lateral, previous_executed_lateral,
    preference, preference_valid, preference_gate, failure, success, dt=.1, max_lateral=.6,
):
    """Return total and the six named reward groups; failure wins over success."""
    scale = float(dt) / .1
    progress = progress_delta
    clear = .03 * scale * torch.clamp((.57 - clearance) / .57, min=0., max=1.).square()
    lateral = .005 * scale * raw_lateral.abs()
    smooth = .01 * (executed_lateral - previous_executed_lateral).abs()
    pref = .0005 * scale * preference_gate.to(progress.dtype) * preference_valid.to(progress.dtype)
    pref = pref * preference * executed_lateral / float(max_lateral)
    dense = progress - clear - lateral - smooth + pref
    total = torch.where(failure, torch.full_like(dense, -20.), torch.where(success, torch.full_like(dense, 2.), dense))
    zero = torch.zeros_like(dense)
    return {
        "total": total, "failure": torch.where(failure, total, zero),
        "success": torch.where(~failure & success, total, zero), "progress": torch.where(failure | success, zero, progress),
        "clearance": torch.where(failure | success, zero, -clear), "lateral": torch.where(failure | success, zero, -lateral),
        "smooth": torch.where(failure | success, zero, -smooth), "preference": torch.where(failure | success, zero, pref),
    }
