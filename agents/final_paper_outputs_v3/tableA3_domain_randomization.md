| Item | Value | Source |
| --- | --- | --- |
| Training task | s_pcr_new | agents/moe_teacher_best_learnedw.pt |
| Evaluation task | s_pcr_line_avoid_basic stage 4 | eval command |
| Eval target speeds | 0.35,0.50,0.60 | run_pcr_main_table_eval command |
| Training target-speed range [m/s] | 0.25-0.65 | HexGround._sample_pcr_new_curriculum |
| Training obstacle rows | 2-5 | HexGround._sample_pcr_new_curriculum / stage layouts |
| Curriculum budget [episodes] | 120000 | HexPCRNewCfg.navigation |
| Episode length [s] | 50 | HexAvoidBasicCfg.env |
| Stage-4 passage width min [m] | 0.75 | HexAvoidBasicCfg.terrain |
| Capsule obstacle radius [m] | 0.15 | HexAvoidBasicCfg.terrain |
| Capsule slots | 13 | HexAvoidBasicCfg.terrain |
| Box slots | 0 | HexAvoidBasicCfg.terrain |
| Wall slots | 0 | HexAvoidBasicCfg.terrain |
| Terrain seed | 7001 | HexAvoidBasicCfg.terrain |
| Obstacle position jitter [m] | 0.06 | HexAvoidBasicCfg.terrain |
| Mirrored layouts | False | HexAvoidBasicCfg.terrain |
| Friction randomization enabled | True | LeggedRobotCfg.domain_rand |
| Friction range | [0.8,1.2] | hex_terrain_config.domain_rand |
| Base-mass randomization enabled | False | LeggedRobotCfg.domain_rand |
| External pushes enabled | False | hex_terrain_config.domain_rand |
| Low-level observation noise enabled | True | LeggedRobotCfg.noise |
| High-level training affordance | actor-only scene GT; camera disabled | agents/moe_teacher_best_learnedw.pt |
| Deployment sensor target | Intel RealSense D435i | agents/moe_teacher_best_learnedw.pt |
| Depth resolution | 1280 x 720 | final Learned-w checkpoint observation contract |
| Depth FOV H x V [deg] | 87 x 58 | final Learned-w checkpoint observation contract |
| Depth range [m] | [0.28,3] | final Learned-w checkpoint observation contract |
| Configured/effective depth rate [Hz] | 30 / 25.0 | final Learned-w checkpoint observation contract |
| High-level rate [Hz] | 10.0 | final Learned-w checkpoint observation contract |
| Affordance map | 32 x 32, extent 3 m | final Learned-w checkpoint observation contract |
