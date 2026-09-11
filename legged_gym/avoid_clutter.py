"""Deterministic s_avoid_clutter geometry, checked in combined circle space."""
from dataclasses import dataclass
import numpy as np

RANGES = {1: ((.20, .30), (.20, .30), (.15, .30), (2.5, 3.), (1, 1)), 2: ((.15, .35), (.15, .30), (.15, .40), (1.8, 3.), (1, 2)), 3: ((.20, .40), (.10, .25), (.15, .50), (1.4, 2.5), (2, 3)), 4: ((.15, .50), (.075, .30), (0., .60), (1.2, 3.), (2, 5))}

@dataclass(frozen=True)
class ClutterLayout:
    stage: int
    kind: str
    decision_episode: bool
    cylinders_xy: np.ndarray
    band_y: np.ndarray
    safe_intervals: tuple
    reachable_intervals: tuple
    exit_y: float
    v_drive_nom: float
    min_forward_assumption: float
    combined_radius: float
    m_clear: float
    kappa: float
    band_enter_y: np.ndarray
    band_exit_y: np.ndarray
    coverage: np.ndarray
    recovery_paths: tuple
    band_decisions: tuple = ()
    speed_draws: int = 1
    layout_attempts: int = 1
    structure_candidates: int = 1

@dataclass(frozen=True)
class BandDecision:
    d_min_m: float
    d_max_m: float
    mandatory: bool
    choice: bool
    effective: bool

def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if lo > hi:
            continue
        if not result or lo > result[-1][1] + 1e-6:
            result.append((float(lo), float(hi)))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], float(hi)))
    return tuple(result)

def propagate(src, dst, budget):
    return union((max(a - budget, c), min(b + budget, d)) for a, b in src for c, d in dst)
propagate_reachable = propagate

def distance_to_union(x, intervals):
    return min(max(lo - x, 0., x - hi) for lo, hi in intervals)

def worst_distance_to_union(interval, targets):
    points = [interval[0], interval[1]]
    for left, right in zip(targets[:-1], targets[1:]):
        midpoint = .5 * (left[1] + right[0])
        if interval[0] <= midpoint <= interval[1]:
            points.append(midpoint)
    return max(distance_to_union(point, targets) for point in points)

def min_distance_to_union(intervals, targets):
    return min(max(target_lo - source_hi, source_lo - target_hi, 0.)
               for source_lo, source_hi in intervals for target_lo, target_hi in targets)

def segment_clear(a, b, center, radius):
    direction = b - a
    length_sq = float(direction @ direction)
    t = 0. if length_sq < 1e-12 else np.clip((center - a) @ direction / length_sq, 0., 1.)
    return np.linalg.norm(a + t * direction - center) > radius + 1e-6

class AvoidClutterGenerator:
    camera_far_m = 3.
    camera_offset_m = .22
    camera_half_fov_rad = np.deg2rad(87. / 2.)
    path_epsilon = .005
    max_dynamic_half_width = 10.
    nominal_traversal_limit_s = 45.
    def __init__(self, radius=.15, strict_radius=.28, band_half_width=1.55, spawn_y=-1.6, v_lat_eff=.15, response_delay=.3, v_lat_max=.6, y_jitter=.04, max_attempts=128, episode_length_s=50):
        self.radius = radius
        self.strict_radius = strict_radius
        self.combined_radius = radius + strict_radius
        self.band_half_width = band_half_width
        self.spawn_y = spawn_y
        self.v_lat_eff = v_lat_eff
        self.response_delay = response_delay
        self.v_lat_max = v_lat_max
        self.y_jitter = y_jitter
        self.max_attempts = max_attempts
        self.episode_length_s = episode_length_s

    def _sample_kind(self, rng):
        draw = rng.uniform()
        return "free" if draw < .10 else ("central" if draw < .20 else "decision")

    def _safe_intervals(self, rng, stage, kind, reference, margin, side_range):
        if kind == "central":
            return ((-margin, margin),)
        if stage <= 2 or rng.uniform() < .5:
            offset = rng.uniform(*side_range)
            return ((reference + offset, reference + offset + 2. * margin),) if rng.uniform() < .5 else ((reference - offset - 2. * margin, reference - offset),)
        center = rng.uniform(-.10, .10); left_margin = rng.uniform(*RANGES[stage][1]); right_margin = rng.uniform(*RANGES[stage][1]); radius = self.combined_radius
        return ((center - radius - 2. * left_margin, center - radius), (center + radius, center + radius + 2. * right_margin))

    def _chain(self, rng, left, right, y):
        length = right - left
        if abs(length) <= 1e-8:
            return [(left, y)]
        if length < 2. * self.radius - 1e-6:
            raise RuntimeError("short non-gap connection")
        pieces = int(np.ceil(length / .60)); spacing = length / pieces
        if spacing < 2. * self.radius - 1e-6:
            raise RuntimeError("overlapping non-gap connection")
        max_x_offset = .45 * min(
            spacing - 2. * self.radius,
            np.sqrt((2. * self.combined_radius) ** 2 - (2. * self.y_jitter) ** 2) - spacing,
        )
        x_offsets = np.zeros(pieces + 1)
        x_offsets[1:-1] = rng.uniform(-max_x_offset, max_x_offset, size=pieces - 1)
        return [
            (left if index == 0 else right if index == pieces else left + index * spacing + x_offsets[index], y if index in (0, pieces) else y + rng.uniform(-self.y_jitter, self.y_jitter))
            for index in range(pieces + 1)
        ]

    def _row(self, rng, gaps, coverage, y):
        points = []; cursor = coverage[0]
        for lo, hi in gaps:
            points.extend(self._chain(rng, cursor, lo - self.combined_radius, y)); cursor = hi + self.combined_radius
        points.extend(self._chain(rng, cursor, coverage[1], y))
        unique = {}
        for x, point_y in points:
            unique.setdefault(round(x, 8), (x, point_y))
        row = np.asarray([unique[key] for key in sorted(unique)], dtype=np.float32)
        for first, second in zip(row[:-1], row[1:]):
            midpoint = .5 * (float(first[0]) + float(second[0])); spacing = float(np.linalg.norm(second - first))
            if not any(lo < midpoint < hi for lo, hi in gaps) and not (2. * self.radius - 1e-6 <= spacing <= 2. * self.combined_radius + 1e-6):
                raise RuntimeError("non-gap cylinder chain invalid")
        return row

    def _coverage(self, reachable, previous_exit, current_exit, speed):
        local_time = (current_exit - previous_exit) / speed + self.response_delay
        extent = self.v_lat_max * local_time + self.combined_radius
        # This is a resource extent, not a geometry rejection criterion.  The
        # generated gaps remain inside this clipped map; no route is placed
        # outside the 10 m local coverage resource.
        return (max(-self.max_dynamic_half_width, min(lo for lo, _ in reachable) - extent),
                min(self.max_dynamic_half_width, max(hi for _, hi in reachable) + extent))

    def _band_free_and_budget(self, speed, kappa, d_max, first):
        """Response slack; the first-band spawn buffer is imposed separately."""
        base = speed * kappa * (self.response_delay + d_max / self.v_lat_eff)
        eps_lo = .03
        eps_hi = min(.12, self.camera_far_m - self.camera_offset_m - 2. * (self.combined_radius + self.y_jitter) - base)
        if eps_lo > eps_hi + 1e-9:
            return None
        return eps_lo, eps_hi

    def _kappa_interval(self, speed, requested_dmax, stage):
        """Analytic domain whose response time remains inside the camera range."""
        lo, hi = RANGES[stage][3]
        factor = speed * (self.response_delay + requested_dmax / self.v_lat_eff)
        if factor <= 0.:
            return lo, hi
        camera_free = self.camera_far_m - self.camera_offset_m - 2. * (self.combined_radius + self.y_jitter)
        return lo, min(hi, (camera_free - .03) / factor)

    def _mandatory_gap(self, reachable, margin, direction, offset):
        """Translate the actual reachable union by its sampled side offset."""
        lo, hi = reachable[0][0], reachable[-1][1]
        if direction > 0:
            return ((hi + offset, hi + offset + 2. * margin),)
        return ((lo - offset - 2. * margin, lo - offset),)

    def _straight_gap(self, reachable, margin):
        """Keep a fixed-width recovery gap around the actual single interval."""
        if len(reachable) != 1 or reachable[0][1] - reachable[0][0] > 2. * margin + 1e-7:
            raise RuntimeError("constructive recovery gap cannot contain reachable union")
        anchor = .5 * (reachable[0][0] + reachable[0][1])
        return ((anchor - margin, anchor + margin),)

    def _choice_gaps(self, reachable, margin):
        """Two same-source gaps around an anchor in the actual, not planned, R."""
        anchor = .5 * (reachable[0][0] + reachable[0][1])
        return ((anchor - .43 - 2. * margin, anchor - .43),
                (anchor + .43, anchor + .43 + 2. * margin))

    def _visible(self, free_distance, reachable, gaps, budget):
        envelope = self.combined_radius + self.y_jitter
        if free_distance > self.camera_far_m - self.camera_offset_m - 2. * envelope:
            return False
        visible_half_width = np.tan(self.camera_half_fov_rad) * (
            self.camera_far_m - self.camera_offset_m - envelope
        )
        offsets = []
        for lo, hi in reachable:
            for x in (lo, .5 * (lo + hi), hi):
                offsets.append(abs(self._nearest_target(x, gaps, budget) - x))
        return max(offsets) + envelope <= visible_half_width

    def _reachable_targets(self, reachable, gaps, budget):
        result = propagate(reachable, gaps, budget)
        if not result or any(not propagate((branch,), gaps, budget) for branch in reachable) or any(not propagate(reachable, (gap,), budget) for gap in gaps):
            return ()
        return result

    def _choice_paths(self, reachable, gaps, budget, previous_exit, enter_y, exit_y):
        """Return same-source, interior-gap witness routes for every valid gap pair."""
        routes = []
        for source_lo, source_hi in reachable:
            for first_index, first in enumerate(gaps[:-1]):
                for second in gaps[first_index + 1:]:
                    first_lo, first_hi = first[0] + self.path_epsilon, first[1] - self.path_epsilon
                    second_lo, second_hi = second[0] + self.path_epsilon, second[1] - self.path_epsilon
                    if first_lo > first_hi or second_lo > second_hi:
                        continue
                    witness_lo = max(source_lo, first_lo - budget, second_lo - budget)
                    witness_hi = min(source_hi, first_hi + budget, second_hi + budget)
                    if witness_lo > witness_hi + 1e-7:
                        continue
                    witness = .5 * (witness_lo + witness_hi)
                    targets = []
                    for gap_lo, gap_hi in ((first_lo, first_hi), (second_lo, second_hi)):
                        targets.append(min(max(witness, gap_lo), gap_hi))
                    routes.extend(((witness, previous_exit), (target, enter_y), (target, exit_y)) for target in targets)
        return tuple(routes)

    def _nearest_target(self, x, gaps, budget):
        candidates = []
        for lo, hi in gaps:
            lo, hi = max(lo, x - budget), min(hi, x + budget)
            if lo <= hi + 1e-7:
                lo_i, hi_i = lo + self.path_epsilon, hi - self.path_epsilon
                target = min(max(x, lo_i), hi_i) if lo_i <= hi_i else .5 * (lo + hi)
                candidates.append((abs(target - x), target))
        if not candidates:
            raise RuntimeError("unrecoverable predecessor point")
        return min(candidates)[1]

    def _recovery_paths(self, reachable, gaps, budget, previous_exit, enter_y, exit_y):
        anchors = [x for lo, hi in reachable for x in (lo, .5 * (lo + hi), hi)]
        paths = [((x, previous_exit), (self._nearest_target(x, gaps, budget), enter_y), (self._nearest_target(x, gaps, budget), exit_y)) for x in anchors]
        for gap in gaps:
            choices = []
            for x in anchors:
                lo, hi = max(gap[0], x - budget), min(gap[1], x + budget)
                if lo <= hi + 1e-7:
                    lo_i, hi_i = lo + self.path_epsilon, hi - self.path_epsilon; target = min(max(x, lo_i), hi_i) if lo_i <= hi_i else .5 * (lo + hi)
                    choices.append((abs(target - x), x, target))
            if not choices:
                raise RuntimeError("next gap has no predecessor path")
            _, x, target = min(choices); path = ((x, previous_exit), (target, enter_y), (target, exit_y))
            if path not in paths:
                paths.append(path)
        return tuple(paths)

    def _paths_clear(self, paths, cylinders):
        return all(segment_clear(np.asarray(start), np.asarray(end), cylinder, self.combined_radius) for path in paths for start, end in zip(path[:-1], path[1:]) for cylinder in cylinders)

    def generate(self, stage, seed, env_id, episode):
        if stage not in RANGES:
            raise ValueError("stage must be 1..4")
        rng = np.random.RandomState((seed + 10007 * env_id + 131 * episode) % (2 ** 32 - 1)); kind = self._sample_kind(rng)
        velocity_range, margin_range, side_range, kappa_range, band_range = RANGES[stage]
        # Speed is deliberately drawn once from the frozen stage range.  All
        # following draws are conditioned on its exact geometric feasibility.
        speed = rng.uniform(*velocity_range)
        if kind == "free":
            return ClutterLayout(stage, kind, False, np.zeros((0, 2), np.float32), np.zeros(0, np.float32), (), (), self.spawn_y + .45, speed, speed, self.combined_radius, 0., 0., np.zeros(0), np.zeros(0), np.zeros((0, 2)), (), (), 1, 1, 1)

        if kind == "central":
            selected_band_count = 1
            required_dmax = 0.
        else:
            if stage <= 2:
                required_dmax = max(.10, side_range[0])
            elif stage == 3:
                required_dmax = .43
            else:
                required_dmax = .10

        k_lo, k_hi = self._kappa_interval(speed, required_dmax, stage)
        if k_lo > k_hi + 1e-9:
            raise RuntimeError("constructive s_avoid_clutter has no kappa domain for sampled speed")
        kappa = rng.uniform(k_lo, k_hi)
        margin = rng.uniform(*margin_range)
        if kind == "central":
            roles = ("straight",)
        else:
            # Every row is conservatively charged with the camera-limited
            # free distance plus its full envelope; this never undercounts a
            # first row or a later mandatory response.
            max_row_distance = (self.camera_far_m - self.camera_offset_m -
                                2. * (self.combined_radius + self.y_jitter) +
                                2. * (self.combined_radius + self.y_jitter))
            max_count = int(np.floor((self.nominal_traversal_limit_s * speed - .45) / max_row_distance))
            feasible_upper = min(band_range[1], max_count)
            if feasible_upper < band_range[0]:
                raise RuntimeError("constructive band-count domain is empty for sampled speed/kappa")
            selected_band_count = rng.randint(band_range[0], feasible_upper + 1)
            # Required roles are first; extra rows are legal recovery rows.
            if stage <= 2:
                roles = ("mandatory",) + ("straight",) * (selected_band_count - 1)
            elif stage == 3:
                roles = ("straight",) * (selected_band_count - 1) + ("choice",)
            else:
                roles = ("mandatory", "mandatory") + ("straight",) * (selected_band_count - 2)

        reachable = ((0., 0.),); previous_exit = self.spawn_y
        rows = []; safe_rows = []; reachable_rows = []; enter_rows = []; exit_rows = []; cover_rows = []; proof_rows = []; decisions = []
        for index, role in enumerate(roles):
            if role == "mandatory":
                camera_free = self.camera_far_m - self.camera_offset_m - 2. * (self.combined_radius + self.y_jitter)
                offset_hi = min(side_range[1], self.v_lat_eff * ((camera_free - .030001) / (speed * kappa) - self.response_delay))
                offset_lo = max(.10, side_range[0])
                if stage == 4:
                    # Two Full mandatory rows use the minimum effective shift:
                    # then the first reachable width bounds the second demand.
                    offset_lo = offset_hi = .10
                if offset_hi < offset_lo - 1e-9:
                    raise RuntimeError("constructive mandatory side-offset domain is empty for sampled speed/kappa")
                offset = rng.uniform(offset_lo, offset_hi)
                gaps = self._mandatory_gap(reachable, margin, 1 if rng.uniform() < .5 else -1, offset)
            elif role == "choice":
                gaps = self._choice_gaps(reachable, margin)
            else:
                gaps = self._straight_gap(reachable, margin)
            d_min = min_distance_to_union(reachable, gaps)
            d_max = max(worst_distance_to_union(branch, gaps) for branch in reachable)
            slack = self._band_free_and_budget(speed, kappa, d_max, first=(index == 0))
            if slack is None:
                raise RuntimeError("constructive timing domain was inconsistent")
            epsilon = rng.uniform(*slack)
            response_free_distance = speed * kappa * (self.response_delay + d_max / self.v_lat_eff) + epsilon
            # The first physical band starts at least 2 m after spawn.  This
            # is geometry, not response slack, so central/straight samples do
            # not need a fictitious epsilon large enough to create it.
            free_distance = max(response_free_distance, 1.53) if index == 0 else response_free_distance
            envelope = self.combined_radius + self.y_jitter
            center_y = previous_exit + free_distance + envelope
            enter_y, exit_y = center_y - envelope, center_y + envelope
            budget = self.v_lat_eff * max(free_distance / speed / kappa - self.response_delay, 0.)
            next_reachable = self._reachable_targets(reachable, gaps, budget)
            if not next_reachable:
                raise RuntimeError("constructive gap unexpectedly unreachable")
            coverage = self._coverage(reachable, previous_exit, exit_y, speed)
            if coverage[0] > min(lo for lo, _ in gaps) - self.combined_radius or coverage[1] < max(hi for _, hi in gaps) + self.combined_radius:
                raise RuntimeError("constructive gap outside capped coverage")
            if not self._visible(free_distance, reachable, gaps, budget):
                raise RuntimeError("constructive gap violates camera budget")
            row = self._row(rng, gaps, coverage, center_y)
            paths = self._recovery_paths(reachable, gaps, budget, previous_exit, enter_y, exit_y)
            choice_paths = self._choice_paths(reachable, gaps, budget, previous_exit, enter_y, exit_y)
            cylinders = np.concatenate(rows + [row])
            choice = bool(choice_paths) and self._paths_clear(choice_paths, cylinders)
            all_paths = tuple(path for proof in proof_rows for path in proof) + paths + (choice_paths if choice else ())
            if not self._paths_clear(all_paths, cylinders):
                raise RuntimeError("constructive recovery proof intersects an expanded cylinder")
            mandatory = d_min >= .10 - 1e-7
            decisions.append(BandDecision(d_min, d_max, mandatory, choice, mandatory or choice))
            rows.append(row); safe_rows.append(tuple(gaps)); reachable_rows.append(next_reachable)
            enter_rows.append(enter_y); exit_rows.append(exit_y); cover_rows.append(coverage)
            proof_rows.append(paths + (choice_paths if choice else ()))
            reachable, previous_exit = next_reachable, exit_y

        final_exit = previous_exit + .45; cylinders = np.concatenate(rows)
        if (final_exit - self.spawn_y) / speed > self.nominal_traversal_limit_s + 1e-7:
            raise RuntimeError("constructive traversal exceeds 45 s")
        if len(cylinders) > 192:
            raise RuntimeError("constructive cylinder count exceeds 192")
        return ClutterLayout(stage, kind, kind == "decision", cylinders,
                             np.asarray([(lo + hi) * .5 for lo, hi in zip(enter_rows, exit_rows)], np.float32),
                             tuple(safe_rows), tuple(reachable_rows), final_exit, speed, speed,
                             self.combined_radius, margin, kappa, np.asarray(enter_rows, np.float32),
                             np.asarray(exit_rows, np.float32), np.asarray(cover_rows, np.float32),
                             tuple(proof_rows), tuple(decisions), 1, 1, selected_band_count)
