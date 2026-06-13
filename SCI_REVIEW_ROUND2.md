# MedMamba Q2-SCI Pre-Submission Code Review -- Round 2

**Review Date:** 2026-05-29
**Round:** 2 (follow-up on Round 1)
**Target:** Q2 SCI Journal (e.g., Computerized Medical Imaging and Graphics, Medical Image Analysis, IEEE JBHI)
**Scope:** Verification of 6 Round-1 bug fixes + discovery of new critical issues

---

## Executive Summary

Round 1 identified **6 bugs** (3 HIGH, 1 MEDIUM, 2 LOW) and assigned a **5.5/10 overall score**. In Round 2, each fix was verified by reading source code, checking import chains, and tracing forward-pass data flow.

**Results of Round-1 bug verification:**
- **3 of 6 bugs fully fixed** (BUG-3, BUG-5, BUG-6)
- **2 of 6 bugs mostly fixed** with residual issues (BUG-2, BUG-4)
- **1 of 6 bugs partially fixed** -- V2 fixed but same bug persists in V3 (BUG-1)
- **4 new bugs discovered**, including **2 CRITICAL** issues that prevent the code from running at all

The most severe finding is **NEW-BUG-1**: a file encoding corruption in `vmamba_blocks.py` (42 null bytes from WSL/UTF-16LE interference) that makes **every model unimportable**. The second critical finding is a channel dimension mismatch between `patch_embed` and `DualBranchEncoder.input_proj` that would crash MedMambaV2 at runtime even if the encoding issue were fixed.

---

## Round-1 Bug Verification (Detailed)

### BUG-1 [CRITICAL]: `cls_token` Dead Code in MedMambaV2 -- PARTIALLY FIXED

**Fix in MedMambaV2:** VERIFIED FIXED.

The dead `cls_token` parameter has been replaced with a SE-style (Squeeze-and-Excitation) channel attention module (`self.global_semantic`) at `src/models/medmamba.py` lines 494-501. The forward pass correctly uses it (lines 533-534):
```python
channel_weights = self.global_semantic(x)  # [B, d_model]
x = x * channel_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
```
This is architecturally sound -- a learnable channel attention that works with `[B, C, H, W]` tensors.

**Same bug persists in MedMambaV3:** NOT FIXED.

`MedMambaV3` at line 640 still declares `self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))`, and `_init_weights` initializes it (line 645). However, the forward pass at lines 675-677 uses `moe_feat.mean(dim=(2, 3), keepdim=True)` instead of `self.cls_token`:
```python
cls_out = moe_feat.mean(dim=(2, 3), keepdim=True)
moe_feat = moe_feat + cls_out
```
The `self.cls_token` parameter is **consumed by the optimizer but never used in computation** -- pure dead code. This wastes memory, inflates parameter counts, and will confuse reviewers.

**Status:** PARTIALLY FIXED (V2 fixed, V3 still broken).

---

### BUG-2 [HIGH]: Duplicate Class Definitions Shadow Imports -- MOSTLY FIXED

**File:** `src/models/medmamba.py`

Round 1 identified 5 duplicate classes. Round 2 verification:

| Class | Was Local? | Current Status | Verdict |
|-------|-----------|----------------|---------|
| `CrossScan` | line 54 | Removed; imported from `vmamba_blocks` (line 24) | FIXED |
| `CrossMerge` | line 76 | Removed; imported from `vmamba_blocks` (line 24) | FIXED |
| `VSSBlock2D` | line 171 | Removed; imported from `vmamba_blocks` (line 24) | FIXED |
| `CTMTrajectoryAnalyzer` | line 232 | Removed; imported from `home_moe` (line 30) | FIXED |
| `RMSNorm` | line 39 | **STILL locally defined** at line 39-47 | NOT FIXED |

`RMSNorm` is defined identically in **four** files:
1. `src/models/medmamba.py` (line 39)
2. `src/models/vmamba_blocks.py` (line 32)
3. `src/models/home_moe.py` (line 31)
4. `src/models/selective_state_space.py` (line 15)

The comment at lines 50-53 correctly states the intent ("统一从 vmamba_blocks.py 导入"), but the local `RMSNorm` definition was not removed. While functionally harmless (all implementations are identical), it violates the DRY principle and the comment's stated intent.

**Status:** MOSTLY FIXED (4/5 duplicates removed).

---

### BUG-3 [HIGH]: MedMambaV3 Forward Loop Incorrect Depth -- FULLY FIXED

**File:** `src/models/medmamba.py`

The `nn.ModuleList` of `VMamba2D(depth=1)` blocks is now correctly used (lines 596-609):
```python
self.vmamba_blocks = nn.ModuleList([
    VMamba2D(d_model=d_model, depth=1, ...)
    for _ in range(n_layers)
])
```

Forward loop iterates exactly `n_layers` times (lines 663-666):
```python
for i in range(len(self.vmamba_blocks)):
    cnn_feat = self.cnn_branch[i](ssm_feat) if i % 2 == 0 else None
    ssm_feat = self.vmamba_blocks[i](ssm_feat, cnn_feat)
```

Network depth is now correctly `n_layers`, not `n_layers * n_layers`.

**Status:** FULLY FIXED.

---

### BUG-4 [MEDIUM]: Training Loop Mixup Type Mismatch -- MOSTLY FIXED

**File:** `train_medmamba.py`

Both AMP (lines 631-635) and non-AMP (lines 652-656) paths now correctly use:
```python
loss_a = self.criterion.task_loss(logits, labels_a)
loss_b = self.criterion.task_loss(logits, labels_b)
loss = lam * loss_a + (1 - lam) * loss_b
```

This bypasses `MedMambaLoss.forward()` (which expects `Dict` input) and directly calls the task loss (`CrossEntropyLoss`).

**Residual issue:** The separate `Trainer` class in `src/trainer.py` (lines 430-446) has a related bug in its mixup path:
```python
outputs = self.model(images)
loss_a = self.criterion(outputs, labels_a)  # criterion is nn.CrossEntropyLoss()
```
If the model returns a `dict` (as all MedMamba models do), `nn.CrossEntropyLoss(outputs, labels)` will crash. However, this class is not used by the main training script (`train_medmamba.py` uses its own `MedMambaTrainer`), so the impact is limited.

**Status:** MOSTLY FIXED (main path fixed, secondary Trainer class still buggy).

---

### BUG-5 [LOW]: Missing `import random` in trainer.py -- FULLY FIXED

**File:** `src/trainer.py` line 19

`import random` is now present at line 19.

**Status:** FULLY FIXED.

---

### BUG-6 [LOW]: Missing `import numpy` in train_medmamba.py -- FULLY FIXED

**File:** `train_medmamba.py` line 37

`import numpy as np` is now present at line 37.

**Status:** FULLY FIXED.

---

## New Bugs Discovered in Round 2

### NEW-BUG-1 [CRITICAL]: `vmamba_blocks.py` Encoding Corruption -- FILE UNIMPORTABLE

**Severity:** CRITICAL -- Blocks all model instantiation
**File:** `src/models/vmamba_blocks.py`

**Problem:**
The file contains **42 null bytes** (`\x00`) caused by UTF-16LE encoding or WSL file system interference. The first 3 lines of the file contain corrupted content:
```
w s l :   h K m 0 R   l o c a l h o s t   ...
```
Each ASCII character is followed by a null byte (UTF-16LE pattern). Python's default UTF-8 codec raises `SyntaxError: source code string cannot contain null bytes` when importing this file.

**Impact:**
- `from .vmamba_blocks import CrossScan, CrossMerge, ...` fails with SyntaxError
- MedMambaV2, MedMambaV3, MedMambaGuard all depend on this import
- **No model can be instantiated or used**
- All tests fail
- benchmark.py fails

**Fix:**
Re-save `vmamba_blocks.py` in UTF-8 encoding. Remove the corrupted first 3 lines (the WSL header content). The actual Python code starting at line 4 (`"""VMamba Blocks...""")` is intact.

**Verification:**
```
$ python -c "
with open('src/models/vmamba_blocks.py', 'rb') as f:
    data = f.read()
    print(f'Null bytes: {data.count(b\"\\x00\")}')  # Output: 42
"
```

---

### NEW-BUG-2 [CRITICAL]: `DualBranchEncoder.input_proj` Channel Mismatch -- Runtime Crash

**Severity:** CRITICAL -- MedMambaV2 will crash on forward pass
**File:** `src/models/medmamba.py` lines 303, 340

**Problem:**
`MedMambaV2` applies `patch_embed` first (converting 3 channels to `d_model`), then passes the result to `DualBranchEncoder`. However, `DualBranchEncoder.__init__` creates:
```python
self.input_proj = nn.Conv2d(3, d_model, kernel_size=1)  # Expects 3 input channels
```

And its forward pass unconditionally applies:
```python
x = self.input_proj(x)  # Line 340 -- CRASH if x has d_model channels
```

**Data flow:**
1. `MedMambaV2.forward`: `x = self.patch_embed(x)` produces `[B, d_model, H/P, W/P]` (e.g., d_model=64)
2. `MedMambaV2.forward`: `x, ctm_score = self.encoder(x)` passes to DualBranchEncoder
3. `DualBranchEncoder.forward`: `x = self.input_proj(x)` -- `Conv2d(3, 64)` receives 64-channel input

**Runtime error:**
```
RuntimeError: Given groups=1, weight of size [64, 3, 1, 1], expected input[1, 64, 8, 8] to have 3 channels, but got 64 channels instead
```

**Impact:** MedMambaV2 cannot execute a forward pass. This means the V2 model has never been successfully tested end-to-end.

**Fix options:**
1. Remove `self.input_proj` from `DualBranchEncoder` and let the caller handle projection
2. Make `input_proj` conditional: `self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=1)` (identity-style)
3. Remove `patch_embed` from `MedMambaV2` and let `DualBranchEncoder` handle it

**Note:** This bug was identified as a weakness in Round 1 ("two redundant projections") but was not flagged as a crash bug. It is actually a **runtime crash**.

---

### NEW-BUG-3 [HIGH]: `MoEEnsemble` Attention Branch Shape Bug -- Incorrect Output

**Severity:** HIGH -- Silent wrong results when `ensemble_method="attention"` is used
**File:** `src/models/home_moe.py` lines 651-658

**Problem:**
In the `MoEEnsemble.forward` method, the attention branch:
```python
stacked = torch.stack(transformed, dim=1)  # [B, num_experts, L, D]
B, n_exp, L, D = stacked.shape
stacked_flat = stacked.reshape(B * n_exp, L, D)

output, _ = self.attention(stacked_flat, stacked_flat, stacked_flat)
output = output.mean(dim=0)  # BUG: collapses batch+expert dimension
```

After `self.attention`, `output` has shape `[B * n_exp, L, D]`. `output.mean(dim=0)` takes the mean over the **entire first dimension** (B * n_exp), producing shape `[L, D]`. This:
- **Loses the batch dimension** entirely
- **Mixes different samples' expert outputs** together
- Returns the wrong shape `[L, D]` instead of `[B, L, D]`

**Correct fix:**
```python
output = output.reshape(B, n_exp, L, D).mean(dim=1)  # [B, L, D]
```

**Status:** NOT FIXED (bug existed in Round 1 as a noted weakness but was not in the 6-bug list).

---

### NEW-BUG-4 [MEDIUM]: `src/trainer.py` Trainer Mixup Path Doesn't Handle Dict Outputs

**Severity:** MEDIUM -- Crashes when Mixup is enabled with MedMamba models
**File:** `src/trainer.py` lines 430-446

**Problem:**
When mixup is active, the `Trainer.train_epoch` method passes the model output directly to the criterion:
```python
outputs = self.model(images)
loss_a = self.criterion(outputs, labels_a)  # criterion = nn.CrossEntropyLoss()
```

If `outputs` is a `dict` (as all MedMamba models return), `nn.CrossEntropyLoss()` will crash because it expects tensor inputs. The standard training path (lines 460-480) correctly handles dict outputs with `outputs.get('logits')`, but the mixup path does not.

**Impact:** The `Trainer` class (separate from `MedMambaTrainer` in `train_medmamba.py`) cannot be used with mixup enabled for MedMamba models. Since the main training script uses its own trainer class, the practical impact is limited.

---

## Summary of All Bugs

| # | Bug | Severity | Round 1 | Round 2 Status |
|---|-----|----------|---------|----------------|
| BUG-1 | cls_token dead code (V2) | CRITICAL | Found | FIXED in V2 |
| BUG-1b | cls_token dead code (V3) | HIGH | -- | STILL PRESENT |
| BUG-2 | Duplicate class definitions | HIGH | Found | MOSTLY FIXED (RMSNorm remains) |
| BUG-3 | V3 forward loop depth | HIGH | Found | FULLY FIXED |
| BUG-4 | Mixup type mismatch | MEDIUM | Found | MOSTLY FIXED (main path fixed) |
| BUG-5 | Missing `import random` | LOW | Found | FULLY FIXED |
| BUG-6 | Missing `import numpy` | LOW | Found | FULLY FIXED |
| NEW-1 | vmamba_blocks.py encoding corruption | CRITICAL | -- | NEW |
| NEW-2 | DualBranchEncoder channel mismatch | CRITICAL | -- | NEW |
| NEW-3 | MoEEnsemble attention shape bug | HIGH | -- | NEW |
| NEW-4 | Trainer mixup dict handling | MEDIUM | -- | NEW |

**Bug fix rate:** 3 fully fixed / 6 identified = 50%. 2 mostly fixed. 1 partially fixed.
**New bugs found:** 4 (2 CRITICAL, 1 HIGH, 1 MEDIUM)

---

## 7-Dimension Scoring (Round 2)

| # | Dimension | R1 Score | R2 Score | Change | Notes |
|---|-----------|:---:|:---:|:---:|-------|
| 1 | **Novelty / Innovation** | 6.5 | 6.5 | -- | No change in novelty claims |
| 2 | **Technical Soundness** | 5.0 | 4.0 | -1.0 | New CRITICAL bugs found (encoding crash, channel mismatch); total bug count increased from 6 to 11 |
| 3 | **Experimental Rigor** | 4.5 | 4.5 | -- | No real experiments added |
| 4 | **Code Quality** | 5.5 | 5.0 | -0.5 | Encoding corruption in vmamba_blocks.py; remaining duplicates; dead code in V3 |
| 5 | **Writing / Presentation** | 6.0 | 6.0 | -- | No changes to documentation |
| 6 | **Significance / Impact** | 6.0 | 6.0 | -- | Conceptual value unchanged |
| 7 | **Reproducibility** | 5.0 | 3.5 | -1.5 | Code cannot run at all due to encoding corruption + channel mismatch; tests cannot execute |

**Round 1 Overall: 5.5 / 10**
**Round 2 Overall: 5.1 / 10 -- DECLINED**

---

## Dimension Analysis Delta (What Changed)

### 2. Technical Soundness (5.0 -> 4.0)

**Improvements from Round 1 fixes:**
- V3 forward loop depth corrected (BUG-3)
- Mixup loss type corrected in main training path (BUG-4)
- Missing imports added (BUG-5, BUG-6)
- SE-style channel attention properly integrated in V2 (BUG-1)

**New issues found:**
- `vmamba_blocks.py` encoding corruption makes the entire model stack unimportable (NEW-BUG-1)
- `DualBranchEncoder.input_proj` channel mismatch would crash MedMambaV2 at runtime (NEW-BUG-2)
- `MoEEnsemble` attention branch silently produces wrong results (NEW-BUG-3)
- `MedMambaV3.cls_token` still dead code (BUG-1b)
- Python `for` loop selective scan still present in both `MambaBlock2D` and `SS2D` (noted in Round 1, unchanged)

**Net assessment:** The Round 1 fixes improved some code paths, but the discovery of 2 new CRITICAL bugs (one of which blocks ALL execution) significantly worsens the technical soundness score.

---

### 7. Reproducibility (5.0 -> 3.5)

**Dropped because:**
- The code **cannot be executed at all** due to `vmamba_blocks.py` encoding corruption
- Even if the encoding were fixed, MedMambaV2 would crash due to channel mismatch
- No tests can pass (test_smoke.py imports vmamba_blocks)
- benchmark.py cannot run
- Docker builds would fail at the import stage

**Unchanged strengths:**
- Docker configuration files exist
- requirements.txt is present
- Test structure is well-designed (but unexecutable)

---

## Priority Action Items (Round 2)

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | **Fix `vmamba_blocks.py` encoding** -- re-save as UTF-8, remove corrupted WSL header | 5 min | CRITICAL: unblocks all execution |
| P0 | **Fix `DualBranchEncoder.input_proj` channel mismatch** -- remove or make conditional | 15 min | CRITICAL: unblocks MedMambaV2 |
| P1 | **Remove dead `cls_token` from MedMambaV3** (line 640, 645) | 5 min | HIGH |
| P1 | **Fix `MoEEnsemble` attention branch** -- reshape before mean | 5 min | HIGH |
| P1 | **Remove duplicate `RMSNorm` from medmamba.py** -- use imported version | 5 min | MEDIUM |
| P2 | **Fix `src/trainer.py` mixup path** -- handle dict outputs | 15 min | MEDIUM |
| P2 | Run `pytest tests/test_smoke.py -v` and fix all failures | 1-2 hours | HIGH |
| P3 | Run experiments on 2+ public medical datasets | 1-2 weeks | CRITICAL for Q2 |
| P3 | Add comparison tables with 5+ baselines | 3-5 days | CRITICAL for Q2 |
| P4 | Translate all Chinese comments to English | 1 day | HIGH |
| P5 | Implement CUDA kernel or use `mamba-ssm` for benchmarks | 3-5 days | HIGH |
| P6 | Add ablation study table | 2-3 days | HIGH |

---

## Files Examined in This Review

| File | Issues Found |
|------|-------------|
| `src/models/vmamba_blocks.py` | NEW-BUG-1: Encoding corruption (42 null bytes) |
| `src/models/medmamba.py` | BUG-1b: V3 cls_token dead code; NEW-BUG-2: DualBranchEncoder channel mismatch; BUG-2 residual: RMSNorm duplicate |
| `src/models/home_moe.py` | NEW-BUG-3: MoEEnsemble attention shape bug |
| `src/models/selective_state_space.py` | Clean -- MambaBlock import works correctly |
| `train_medmamba.py` | BUG-4: FIXED; BUG-6: FIXED |
| `src/trainer.py` | BUG-5: FIXED; NEW-BUG-4: mixup dict handling |
| `benchmark.py` | Cannot run due to vmamba_blocks.py encoding |
| `tests/test_smoke.py` | Cannot run due to vmamba_blocks.py encoding |
| `tests/test_comprehensive.py` | Cannot run due to vmamba_blocks.py encoding |

---

## Conclusion

The Round 1 review identified 6 bugs and the team correctly fixed 3 of them fully and 2 mostly. However, Round 2 discovered **4 new bugs**, including **2 CRITICAL issues** that are more severe than any Round 1 bug:

1. **File encoding corruption** in `vmamba_blocks.py` makes the entire codebase unrunnable
2. **Channel dimension mismatch** between `patch_embed` and `DualBranchEncoder.input_proj` would crash MedMambaV2 even if the encoding were fixed

These findings suggest that the code has **never been executed end-to-end** -- neither the models, nor the tests, nor the benchmarks have been successfully run. This is consistent with Round 1's observation that "no actual training scripts with real datasets" exist.

**For Q2 acceptance, the minimum requirements are:**
1. Fix all P0/P1 bugs (estimated: 30 minutes)
2. Ensure `pytest tests/test_smoke.py` passes (estimated: 1-2 hours)
3. Run experiments on 2+ public datasets with comparison tables (estimated: 1-2 weeks)
4. Translate all documentation to English (estimated: 1 day)

Without steps 1-2, the paper cannot be submitted. Without steps 3-4, it will likely be rejected even at Q2 level.

**Round 2 Verdict: NOT READY for submission. Requires at minimum P0 fixes before any further review.**
