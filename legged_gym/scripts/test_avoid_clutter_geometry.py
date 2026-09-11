#!/usr/bin/env python3
"""One 1024-layout CPU check for the s_avoid_clutter geometry contract."""
import json
import statistics
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legged_gym.avoid_clutter import AvoidClutterGenerator, BandDecision, RANGES, min_distance_to_union, propagate_reachable, segment_clear, worst_distance_to_union

EPS = 1e-5
def assert_chain_x_jitter(generator):
    left, right, y = -1.8, 1.8, 1.0
    chain = np.asarray(generator._chain(np.random.RandomState(41), left, right, y))
    repeat = np.asarray(generator._chain(np.random.RandomState(41), left, right, y))
    alternate = np.asarray(generator._chain(np.random.RandomState(42), left, right, y))
    assert np.array_equal(chain, repeat) and not np.array_equal(chain, alternate)
    assert np.array_equal(chain[[0, -1]], np.asarray(((left, y), (right, y))))
    assert not np.allclose(np.diff(chain[:, 0]), np.diff(chain[:, 0])[0])
    for first, second in zip(chain[:-1], chain[1:]):
        assert 2. * generator.radius - EPS <= np.linalg.norm(second - first) <= 2. * generator.combined_radius + EPS
    gaps = ((-.4, .4),)
    row = generator._row(np.random.RandomState(41), gaps, (-1.8, 1.8), y)
    alternate_row = generator._row(np.random.RandomState(42), gaps, (-1.8, 1.8), y)
    assert not np.array_equal(row, alternate_row)
    assert not np.isclose(row[1, 0] - row[0, 0], row[2, 0] - row[1, 0])
    assert not np.isclose(row[-2, 0] - row[-3, 0], row[-1, 0] - row[-2, 0])

def assert_layout(layout, generator):
    assert layout.exit_y > generator.spawn_y
    speed_range = generator_range = None
    # Stage ranges are checked numerically here rather than trusting layout metadata.
    from legged_gym.avoid_clutter import RANGES
    speed_range = RANGES[layout.stage][0]
    assert speed_range[0] <= layout.v_drive_nom <= speed_range[1]
    if layout.kind == "free":
        assert not layout.decision_episode and not len(layout.safe_intervals) and not layout.band_decisions
        return
    assert layout.decision_episode == (layout.kind == "decision")
    assert len(layout.band_decisions) == len(layout.safe_intervals)
    assert np.isclose(layout.combined_radius, .43) and len(layout.cylinders_xy) <= 192
    margin_range, kappa_range = RANGES[layout.stage][1], RANGES[layout.stage][3]
    assert margin_range[0] <= layout.m_clear <= margin_range[1]
    assert kappa_range[0] <= layout.kappa <= kappa_range[1]
    assert (layout.exit_y - generator.spawn_y) / layout.v_drive_nom <= 45. + EPS
    assert all(np.linalg.norm(a - b) >= 2. * generator.radius - EPS for i, a in enumerate(layout.cylinders_xy) for b in layout.cylinders_xy[i + 1:])
    assert layout.band_y[0] - generator.spawn_y >= 2.0 - EPS
    predecessor = ((0., 0.),); previous_exit = generator.spawn_y
    for index, (gaps, reached, paths, decision) in enumerate(zip(layout.safe_intervals, layout.reachable_intervals, layout.recovery_paths, layout.band_decisions)):
        assert all(np.isclose(hi - lo, 2. * layout.m_clear, atol=EPS) for lo, hi in gaps)
        budget = generator.v_lat_eff * max((layout.band_enter_y[index] - previous_exit) / layout.v_drive_nom / layout.kappa - generator.response_delay, 0.)
        expected_reached = propagate_reachable(predecessor, gaps, budget)
        assert len(reached) == len(expected_reached) and np.allclose(reached, expected_reached, atol=2e-5)
        assert max(worst_distance_to_union(branch, gaps) for branch in predecessor) <= budget + EPS
        assert np.isclose(decision.d_min_m, min_distance_to_union(predecessor, gaps), atol=EPS)
        assert np.isclose(decision.d_max_m, max(worst_distance_to_union(branch, gaps) for branch in predecessor), atol=EPS)
        assert decision.mandatory == (decision.d_min_m >= .10 - EPS)
        assert decision.effective == (decision.mandatory or decision.choice)
        assert all(propagate_reachable((branch,), gaps, budget) for branch in predecessor)
        assert all(propagate_reachable(predecessor, (gap,), budget) for gap in gaps)
        local_time = (layout.band_exit_y[index] - previous_exit) / layout.v_drive_nom + generator.response_delay
        expected = (max(-generator.max_dynamic_half_width, min(lo for lo, _ in predecessor) - generator.v_lat_max * local_time - .43),
                    min(generator.max_dynamic_half_width, max(hi for _, hi in predecessor) + generator.v_lat_max * local_time + .43))
        assert np.allclose(layout.coverage[index], expected, atol=2e-5) and max(abs(v) for v in layout.coverage[index]) <= 10. + EPS
        row = layout.cylinders_xy[np.abs(layout.cylinders_xy[:, 1] - layout.band_y[index]) <= generator.y_jitter + EPS]
        for lo, hi in gaps:
            assert any(abs(point[0] - (lo - .43)) < 2e-5 and abs(point[1] - layout.band_y[index]) < EPS for point in row)
            assert any(abs(point[0] - (hi + .43)) < 2e-5 and abs(point[1] - layout.band_y[index]) < EPS for point in row)
        ordered = row[np.argsort(row[:, 0])]
        for first, second in zip(ordered[:-1], ordered[1:]):
            midpoint = .5 * (first[0] + second[0])
            if not any(lo < midpoint < hi for lo, hi in gaps):
                assert 2. * generator.radius - EPS <= np.linalg.norm(second - first) <= .86 + EPS
        seen = set()
        for path in paths:
            for gap_number, (lo, hi) in enumerate(gaps):
                if lo + EPS <= path[-1][0] <= hi - EPS: seen.add(gap_number)
            assert all(segment_clear(np.asarray(a), np.asarray(b), cylinder, .43) for a, b in zip(path[:-1], path[1:]) for cylinder in layout.cylinders_xy)
        assert seen == set(range(len(gaps)))
        predecessor = reached; previous_exit = layout.band_exit_y[index]

def assert_choice_witness_contract(generator):
    assert min_distance_to_union(((-1., 1.),), ((-.1, .1),)) == 0.
    assert min_distance_to_union(((-1., -.1),), ((-.2, .2),)) == 0.
    assert np.isclose(min_distance_to_union(((-1., -.5),), ((-.3, .2),)), .2)
    true_paths = generator._choice_paths(((0., 0.),), ((-.60, -.20), (.20, .60)), .60, -1., 0., .86)
    assert len(true_paths) == 2 and np.isclose(true_paths[0][0][0], true_paths[1][0][0])
    assert not np.isclose(true_paths[0][-1][0], true_paths[1][-1][0])
    false_paths = generator._choice_paths(((-.50, -.40), (.40, .50)), ((-.60, -.30), (.30, .60)), .15, -1., 0., .86)
    assert not false_paths  # Different predecessor branches are not a choice witness.
    d_min = min_distance_to_union(((0., 0.),), ((-.60, -.20), (.20, .60)))
    both = BandDecision(d_min, .60, d_min >= .10, bool(true_paths), d_min >= .10 or bool(true_paths))
    assert both.mandatory and both.choice and both.effective
    assert sum((both.effective,)) == 1  # Mandatory plus choice is one effective band.

def _summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None

def main():
    generator = AvoidClutterGenerator(); assert_chain_x_jitter(generator); assert_choice_witness_contract(generator); kinds = set(); saw_decision = False; saw_two_gap = False; stages = {}
    for stage in range(1, 5):
        records = {"decision": [], "free_central": []}
        for episode in range(1, 257):
            try:
                layout = generator.generate(stage, 9173, 3, episode); repeat = generator.generate(stage, 9173, 3, episode)
            except RuntimeError as error:
                raise RuntimeError("stage=%d env_id=3 episode=%d seed=9173: %s" % (stage, episode, error)) from error
            assert np.array_equal(layout.cylinders_xy, repeat.cylinders_xy) and layout.kind == repeat.kind
            assert_layout(layout, generator); kinds.add(layout.kind); saw_decision |= layout.decision_episode; saw_two_gap |= any(len(gaps) == 2 for gaps in layout.safe_intervals)
            if stage <= 2: assert all(len(gaps) == 1 for gaps in layout.safe_intervals)
            if layout.kind == "decision":
                lower, upper = RANGES[stage][4]
                assert lower <= len(layout.band_decisions) <= upper
            category = "decision" if layout.decision_episode else "free_central"
            previous = generator.spawn_y
            records[category].append({
                "band_count": len(layout.band_decisions),
                "effective_count": sum(item.effective for item in layout.band_decisions),
                "mandatory_count": sum(item.mandatory for item in layout.band_decisions),
                "choice_count": sum(item.choice for item in layout.band_decisions),
                "d_min_m": [item.d_min_m for item in layout.band_decisions],
                "d_max_m": [item.d_max_m for item in layout.band_decisions],
                "row_spacing_m": [float(enter - prior) for enter, prior in zip(layout.band_enter_y, [previous] + list(layout.band_exit_y[:-1]))],
            })
        stage_summary = {}
        for category, entries in records.items():
            flattened = lambda key: [value for entry in entries for value in entry[key]]
            stage_summary[category] = {
                "layout_count": len(entries), "band_count": _summary([entry["band_count"] for entry in entries]),
                "effective_count": _summary([entry["effective_count"] for entry in entries]),
                "mandatory_count": _summary([entry["mandatory_count"] for entry in entries]),
                "choice_count": _summary([entry["choice_count"] for entry in entries]),
                "d_min_m": _summary(flattened("d_min_m")), "d_max_m": _summary(flattened("d_max_m")),
                "row_spacing_m": _summary(flattened("row_spacing_m")),
            }
        decision = records["decision"]
        if stage <= 2: assert all(entry["mandatory_count"] >= 1 for entry in decision)
        if stage == 3: assert all(entry["choice_count"] >= 1 for entry in decision)
        if stage == 4: assert all(entry["effective_count"] >= 2 for entry in decision)
        stages[str(stage)] = stage_summary
    assert {"free", "central", "decision"} <= kinds and saw_decision and saw_two_gap
    assert len(propagate_reachable(((-1., -.5), (.5, 1.)), ((-.9, -.6), (.6, .9)), .1)) == 2
    assert worst_distance_to_union((-.5, .5), ((-1., -.8), (.8, 1.))) == .8
    output = ROOT / "outputs" / "avoid_clutter_effective_decisions" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seed": 9173, "env_id": 3, "layouts": 1024, "stages": stages}, indent=2) + "\n")
    print("avoid clutter geometry: PASS", output)
if __name__ == "__main__": main()
