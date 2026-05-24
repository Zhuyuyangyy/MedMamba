# MedMamba-Guard V0.2 Smoke Experiment Summary
Generated: 2026-05-17 01:38:34

## Basic Stats
- Total samples: 20
- Overall accuracy: 45.00%

## By Condition

| Condition | Count | Accuracy | Confidence | R_total | R_state | R_scan | R_task | Review Rate |
|-----------|-------|----------|------------|---------|---------|--------|--------|-------------|
| clean | 5 | 80.0% | 0.577 | 0.702 | 1.000 | 0.993 | 0.223 | 100.0% |
| blur | 5 | 20.0% | 0.585 | 0.699 | 1.000 | 0.993 | 0.215 | 0.0% |
| noise | 5 | 60.0% | 0.571 | 0.704 | 1.000 | 0.994 | 0.229 | 100.0% |
| conflict | 5 | 20.0% | 0.577 | 0.702 | 1.000 | 0.993 | 0.223 | 100.0% |

## Key Observations

- ⚠️ Clean samples do NOT have lowest R_total
- ✅ Blur/Noise samples have higher R_total
- ⚠️ Conflict samples do NOT have higher R_task
- ✅ All risks are finite (no NaN/Inf)
- ✅ Gate actions valid