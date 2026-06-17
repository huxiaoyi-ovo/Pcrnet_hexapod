| Method | C_avoid Rate | CSI@C_avoid | Δy_w@C_avoid | Δy_r@C_avoid | Notes |
| --- | --- | --- | --- | --- | --- |
| Y-only | 0.114 | 0.000 | N/A | N/A | No arbitration channel |
| Geom-w | 0.099 | 0.001 | N/A | N/A | Hand-designed geometric w |
| Risk-only | 0.126 | 0.000 | N/A | 5.2e-06 | Trained from scratch with Δy_r only; no learned-w channel |
| Learned-w | 0.135 | -0.065 | 0.0651 | 0.0000 | Learned conflict-conditioned w channel |
