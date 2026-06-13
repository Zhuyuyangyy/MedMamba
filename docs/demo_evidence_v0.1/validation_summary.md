# MedMamba-Guard V0.1 Validation Summary

**Date**: 2026-05-17
**Environment**: Windows / Python 3.12.10 / PyTorch 2.11.0+cpu
**Model Config**: d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4

---

## V0.1 Acceptance Criteria (5 Items)

| # | Acceptance Item | Standard | Result | Evidence |
|---|----------------|----------|--------|----------|
| 1 | Import | All core modules can be imported normally | **PASS** | test_forward.py: all 6 modules imported OK |
| 2 | Forward | Random tensor completes one full inference pass | **PASS** | test_forward.py: output fields complete (prediction, confidence, risk_score, risk_heatmap, review_regions, audit_log) |
| 3 | CTM | Smooth state trajectory → low risk; Jump state → high risk | **PASS** | test_forward.py: smooth V_norm=0.1744 < jump V_norm=9.3994; pytest: test_smooth_vs_jump_sequence PASSED |
| 4 | Cross-Scan | Consistent direction → low risk; Divergent direction → high risk | **PASS** | test_forward.py: consistent risk_l2=0.5000 < divergent risk_l2=1.0000; pytest: test_consistent_features_low_risk + test_divergent_features_high_risk PASSED |
| 5 | Gate | Classification-segmentation conflict triggers doctor_review / overconfidence_warning | **PASS** | test_forward.py: high_conf+no_lesion → REVIEW; high_conf+high_state_risk → doctor_review; pytest: test_high_confidence_no_lesion + test_hard_gating_rules_doctor_review PASSED |

**Verdict: 5/5 PASSED — Trustworthy Inference Logic Closed Loop Complete**

---

## Test Execution Results

### test_forward.py (Interactive Chain Validation)

```
[PASS] import       — All 6 core modules imported
[PASS] forward      — Full inference output with 8 fields
[PASS] ctm_hook     — Hook captured hidden states, CTM metrics computed
[PASS] risk_logic   — CTM/Cross-Scan/Gate/Overconfidence all validated

Total: 4/4 passed (5.0s)
```

### pytest test_guard_core.py (Regression Test Suite)

```
TestCTMMonitor (3/3)
  test_smooth_vs_jump_sequence          PASSED
  test_ctm_metrics_nonempty             PASSED
  test_overconfidence_detection         PASSED

TestCrossScanRiskAnalyzer (5/5)
  test_consistent_features_low_risk     PASSED
  test_divergent_features_high_risk     PASSED
  test_risk_l2_vs_cos_combination       PASSED
  test_get_scan_risk_score              PASSED
  test_get_inconsistent_regions         PASSED

TestTaskConflictValidator (6/6)
  test_high_confidence_large_lesion     PASSED
  test_low_confidence_no_lesion         PASSED
  test_high_confidence_no_lesion        PASSED
  test_low_confidence_large_lesion      PASSED
  test_conflict_score_calculation       PASSED
  test_seg_evidence_breakdown           PASSED

TestMedMambaGuardIntegration (8/8)
  test_forward_output_schema            PASSED
  test_risk_score_range                 PASSED
  test_risk_heatmap_shape               PASSED
  test_review_regions_structure         PASSED
  test_audit_log_contains_ctm           PASSED
  test_hard_gating_rules_doctor_review  PASSED
  test_hard_gating_rules_low_risk_passes PASSED
  test_batch_processing                 PASSED

Total: 22/22 passed (7.40s)
```

---

## Core Module Inventory

1. **CTM State Trajectory Monitor** (`ctm_monitor.py`)
   - V_norm: state transition instability
   - D_delta: input response drift
   - C_layer: trajectory convergence
   - R_overconf: overconfidence risk
   - R_state: composite state risk

2. **Cross-Scan Consistency Risk Analyzer** (`cross_scan_risk.py`)
   - risk_l2: L2 divergence between scan directions
   - risk_cos: cosine divergence between scan directions
   - risk_fused: weighted combination
   - Inconsistent region detection

3. **Classification-Segmentation Conflict Validator** (`task_conflict_validator.py`)
   - Area score / Compactness score / Boundary stability
   - Conflict direction classification
   - Action mapping: PASS / REVIEW

4. **Hard-Gated Audit Inference Pipeline** (`medmamba_guard.py`)
   - R_total = w_ctm * R_state + w_scan * R_scan + R_task
   - Hard gating: doctor_review / overconfidence_warning / PASS
   - Audit logging with full traceability
   - Risk heatmap generation

---

## V0.1 Status

```
MedMamba-Guard V0.1

Status:
- Core trustworthy inference modules implemented.
- AST validation passed for 15/15 Python files.
- Interactive validation script completed (4/4 PASS).
- Pytest-based core unit test suite completed (22/22 PASS).
- CPU runtime validation passed (no GPU required).

Core modules:
1. CTM state trajectory monitor
2. Cross-Scan consistency risk analyzer
3. Classification-segmentation conflict validator
4. Hard-gated audit inference pipeline
```

---

## Next Step: V0.2 Smoke Experiment

Target: 20-sample risk-error correlation experiment

| sample_id | pred | label | correct | confidence | R_state | R_scan | R_task | R_total | gate_action |
|-----------|------|-------|---------|------------|---------|--------|--------|---------|-------------|

Core hypothesis to validate:
> Error samples, ambiguous samples, and classification-segmentation conflict samples have higher risk scores.
