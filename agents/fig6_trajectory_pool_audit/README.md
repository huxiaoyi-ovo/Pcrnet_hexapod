# Fig.6 Trajectory Pool Audit

## Method Summary

| speed | method | episode_count | success | collision | target_lost | timeout | follow_lost | lost_timeout_total | layout_count | layout_sides |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.60 | yonly | 64 | 28 | 24 | 0 | 0 | 12 | 12 | 64 | left_heavy:32,right_heavy:32 |
| 0.60 | geomw | 64 | 47 | 12 | 0 | 0 | 5 | 5 | 64 | left_heavy:32,right_heavy:32 |
| 0.60 | risk_only | 64 | 32 | 9 | 0 | 0 | 23 | 23 | 64 | left_heavy:32,right_heavy:32 |
| 0.60 | rule_override | 64 | 34 | 21 | 0 | 0 | 9 | 9 | 64 | left_heavy:32,right_heavy:32 |
| 0.60 | learnedw | 64 | 62 | 2 | 0 | 0 | 0 | 0 | 64 | left_heavy:32,right_heavy:32 |

## Top Fig.6 Candidate Sets

| score | mirror_ok | baseline_collisions | baseline_lost_timeout | layout_sides | selected_terminations | selected_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| 8950.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:follow_lost,risk_only:collision,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:1,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:1,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:11,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:19,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:23,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:27,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:5,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:collision,rule_override:follow_lost,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:1,rule_override:4604e5c7c1:51,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:13,rule_override:4604e5c7c1:1,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:13,rule_override:4604e5c7c1:13,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:13,rule_override:4604e5c7c1:21,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:13,rule_override:4604e5c7c1:25,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:13,rule_override:4604e5c7c1:3,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:19,rule_override:4604e5c7c1:1,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:19,rule_override:4604e5c7c1:13,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:19,rule_override:4604e5c7c1:21,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:19,rule_override:4604e5c7c1:25,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:19,rule_override:4604e5c7c1:3,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:21,rule_override:4604e5c7c1:1,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:21,rule_override:4604e5c7c1:13,learnedw:306fe51644:1 |
| 8550.000 | 1 | 3 | 1 | left_heavy | yonly:collision,geomw:collision,risk_only:follow_lost,rule_override:collision,learnedw:success | yonly:9bf76297e5:1,geomw:deb7a4ce87:23,risk_only:ca30bd7927:21,rule_override:4604e5c7c1:21,learnedw:306fe51644:1 |

## Run Count

- Runs scanned: 97
