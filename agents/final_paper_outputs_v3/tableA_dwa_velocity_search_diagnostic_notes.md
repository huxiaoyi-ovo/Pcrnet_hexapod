# DWA-style velocity-search diagnostic notes

The table reports bounded validation search biases for the dynamic-window Target-aware Velocity-Space Search baseline. Each row enables the dynamic-window candidate filter before short-horizon rollout and footprint checking. The safety-biased setting suppresses row progress without achieving reliable safety, while the tracking-biased setting improves progress but still incurs frequent collisions. This baseline is therefore treated as a diagnostic planner-style external alternative, not as a main-table competitor.
