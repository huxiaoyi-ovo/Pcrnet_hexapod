# Real-Robot PCR Arbitration Figure Notes

## Source

- Bag: `/home/hxy/下载/pcr_real_20260618_143745.bag`
- SHA256: `ebfc8bb7ab78334e31c68e98d60d634917a8d8fef0ef552fccd6b3abbae89162`
- Debug topic: `/pcr_realplay/debug`
- Selected interval: `114.087--122.284 s` from bag debug start.
- Selected samples: `83` at approximately `10 Hz`.

## Deterministic Selection

- Only continuous samples with valid target, valid depth, no safety stop, and `risk_blocked_map` were eligible.
- Fixed event requirements: current `pcr_real_20260618_*.bag`, risk peak >= 0.45, risk rise >= 0.25, post-event risk recovery >= 0.15, lateral increase >= 0.05 m/s, event forward command >= 0.15 m/s, moving fraction >= 0.80, and arbitration reconstruction error < 1e-5.
- Winning event score: `7.2958`.
- Risk pre/peak/post: `0.0531 / 0.6232 / 0.1012`.
- Lateral magnitude pre/event-p90/post: `0.1930 / 0.2807 / 0.1871` m/s.
- Lateral increase / event forward command: `+0.0877 / 0.4840` m/s.
- Event learned correction mean: `0.0581`.

## Arbitration Reconstruction

- `signed_w = 2w - 1`; values with `|signed_w| <= 0.050` are set to zero.
- `Delta y_r = 0.150 (risk_A - risk_F)`.
- `Delta y_w = 0.300 signed_w_active`.
- `y_eff = clip(y + Delta y_r + Delta y_w, 0, 1)`.
- Reconstruction absolute error mean/max: `2.033e-08 / 5.178e-08`.

## Figure Evidence

- Panel (a) omits the constant-zero `risk_A` curve. In all valid samples of the selected interval, `clearance_A = 3.0 m`, `risk_A_raw = 0`, and `risk_A = 0` because the pure-lateral Avoid cone contains no observed blocked cell.
- The analytic correction in this interval is therefore `Delta y_r = -0.15 risk_F`; this is a recorded property of the deployable local-map risk estimate, not a manually imposed plotting assumption.
- Panel (c) reports `Delta |v_x|(t) = |v_x(t)| - median_pre(|v_x|)`; the pre-conflict window is 2.5--0.35 s before the first high-risk event.
- The recorded pre-conflict lateral baseline is `0.1896 m/s`; the plotted transformation changes only the display origin, while the CSV retains the original signed `cmd_safe_x`.
- The displayed lateral modulation range is `-0.0172` to `+0.0965 m/s`.
- Light-red vertical shading denotes high-risk conflict states and uses recorded `risk_F >= 0.25`, `risk_F > risk_A`, and recorded `conflict_score >= 0.10`. The light-blue area in panel (b) is reserved exclusively for the learned correction `Delta y_w`.
- High-risk spans in the selected interval: `2.90--4.20 s, 6.49--6.79 s, 7.10--8.01 s`.
- During high-risk samples, mean `risk_F`, `Delta y_w`, lateral command, and forward command are `0.5361`, `0.0516`, `-0.2131`, and `0.5228`.
- During high-risk samples, learned-w restores `y_eff - y_risk = 0.0516` on average.

## Claim Boundary

- Supported: the real D435i-derived Follow-command risk changes online, PCR changes the effective Follow weight, and the published command increases its lateral magnitude while retaining forward authority.
- Avoid-risk scope: the observed lateral Avoid cone remains free in this interval, so the trace demonstrates Follow-risk-driven arbitration rather than two dynamically varying command risks.
- Not supported by this trace alone: trial success rate, collision rate, final-row clearance, or statistical sim-to-real superiority. Those require synchronized video and manual trial labels.

## Candidate Audit

| Rank | Start [s] | End [s] | Eligible | Score | Risk peak | Risk range | Mean abs(lat) | Mean fwd |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| 1 | 114.09 | 122.28 | yes | 7.296 | 0.623 | 0.570 | 0.281 | 0.484 |
| 2 | 80.77 | 91.37 | yes | 6.111 | 0.696 | 0.471 | 0.283 | 0.586 |
