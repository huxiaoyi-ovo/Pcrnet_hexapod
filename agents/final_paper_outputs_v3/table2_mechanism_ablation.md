| Method | C_avoid Rate | CSI@C_avoid | Δy_total@C_unsafe | Params | Notes |
| --- | --- | --- | --- | --- | --- |
| Y-only | 0.114 | 0.000 | N/A | 2058890 | No w channel |
| Geom-w | 0.099 | 0.001 | N/A | 2058890 | Geometric w; no learned-w channel |
| Risk-only | 0.126 | 0.000 | 3.6e-06 | 2058762 | Trained from scratch with risk-difference term Δy_r only; no learned-w channel |
| Learned-w | 0.135 | -0.065 | 0.0475 | 2092812 | Learned signed-w channel with analytic risk-difference term |
