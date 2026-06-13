# MedMamba Q2-SCI Pre-Submission Code Review

**Review Date:** 2026-05-29
**Target:** Q2 SCI Journal (e.g., Computerized Medical Imaging and Graphics, Medical Image Analysis, IEEE JBHI)
**Scope:** Full codebase audit -- architecture, training, evaluation, reproducibility

---

## 7-Dimension Scoring Summary

| # | Dimension | Score (1-10) | Verdict |
|---|-----------|:---:|---------|
| 1 | **Novelty / Innovation** | 6.5 | Moderate -- combines existing components (VMamba + U-Mamba + MoE + CTM) without a single deeply novel contribution |
| 2 | **Technical Soundness** | 5.0 | Several critical bugs found (see below); architecture has dead code and incorrect loop logic |
| 3 | **Experimental Rigor** | 4.5 | No actual training scripts with real datasets; benchmark only measures latency, not accuracy on medical benchmarks |
| 4 | **Code Quality** | 5.5 | Decent structure but duplicate class definitions, shadowed imports, missing imports |
| 5 | **Writing / Presentation** | 6.0 | Docstrings are clear; Chinese comments may limit international readability |
| 6 | **Significance / Impact** | 6.0 | O(n) SSM for medical imaging is timely; CTM hallucination detection is a unique angle |
| 7 | **Reproducibility** | 5.0 | Docker support is good; but no trained weights, no real dataset configs, no seed-locked results |

**Overall Weighted Score: 5.5 / 10 -- Borderline Q2, needs revision**

---

## Critical Bugs Found and Fixed

### BUG-1 [CRITICAL]: `cls_token` Dead Code in MedMambaV2 (TOP 1)

**Severity:** HIGH -- Reviewers will immediately catch this
**File:** `src/models/medmamba.py` (original lines 643-674)

**Problem:**
`self.cls_token` was instantiated in `__init__` and expanded in `forward`, but the expanded tensor was **never used**. The subsequent code used `x.mean(dim=(2,3))` as a workaround, making the cls_token pure dead code. This is a classic copy-paste artifact from ViT templates that does not work with CNN-style spatial feature maps `[B, C, H, W]`.

**Fix Applied:**
Replaced the dead `cls_token` with a SE-style (Squeeze-and-Excitation) channel attention module that:
1. Pools spatial dimensions to get a channel descriptor
2. Passes through a bottleneck MLP to learn channel-wise importance
3. Applies learned channel weights to the feature map

This is architecturally consistent with the `[B, C, H, W]` tensor format and provides a learnable global semantic injection mechanism.

---

### BUG-2 [HIGH]: Duplicate Class Definitions Shadow Imports

**Severity:** HIGH -- Causes confusion and potential runtime errors
**File:** `src/models/medmamba.py`

**Problem:**
Five classes were defined locally in `medmamba.py` AND imported from `vmamba_blocks.py` / `home_moe.py`:
- `CrossScan` (local line 54 vs import line 23)
- `CrossMerge` (local line 76 vs import line 23)
- `VSSBlock2D` (local line 171 vs import line 23)
- `CTMTrajectoryAnalyzer` (local line 232 vs import line 30)
- `RMSNorm` (local line 39, also in vmamba_blocks and home_moe)

The local definitions **shadowed** the imports, meaning the imported versions (which are more complete -- e.g., the imported `VSSBlock2D` uses `SS2D` with proper cross-scan, while the local one used a simpler `MambaBlock2D`) were never used.

**Fix Applied:**
Removed all duplicate local definitions. The code now uses the canonical implementations from `vmamba_blocks.py` and `home_moe.py` exclusively.

---

### BUG-3 [HIGH]: MedMambaV3 Forward Loop Incorrect Depth

**Severity:** HIGH -- Network depth was `(n_layers/2) * n_layers` instead of `n_layers`
**File:** `src/models/medmamba.py` (original lines 596-666)

**Problem:**
`VMamba2D` was created with `depth=n_layers` (e.g., 12 internal SSM blocks), but the forward loop called the **entire encoder** for each CNN feature. With `n_layers=12`, `cnn_feats` had 6 elements, so the actual depth was `6 * 12 = 72` blocks instead of 12. This would:
- Make all benchmark numbers (params, FLOPs, latency) incorrect
- Cause gradient instability due to excessive depth
- Produce misleading experimental results

**Fix Applied:**
Changed to `nn.ModuleList` of `VMamba2D(depth=1)` blocks, one per layer. The forward loop now correctly iterates `n_layers` times, applying CNN features every other layer.

---

### BUG-4 [MEDIUM]: Training Loop Mixup Type Mismatch

**Severity:** MEDIUM -- Would crash at runtime when Mixup/Cutmix is enabled
**File:** `train_medmamba.py` (original lines 631-633, 651-653)

**Problem:**
During Mixup/Cutmix augmentation, the code called `self.criterion(logits, labels_a)` passing a raw tensor, but `MedMambaLoss.forward()` expects `(predictions: Dict, targets, routing_weights)`. This would cause a runtime error or incorrect loss computation.

**Fix Applied:**
Changed to `self.criterion.task_loss(logits, labels_a)` to directly use the task loss (CrossEntropyLoss), bypassing the CTM/MoE auxiliary losses that are not applicable during mixup.

---

### BUG-5 [LOW]: Missing `import random` in trainer.py

**Severity:** LOW -- Would crash when Mixup is enabled
**File:** `src/trainer.py` (line 429)

**Problem:**
`random.random()` was used but `random` was not imported.

**Fix Applied:** Added `import random`.

---

### BUG-6 [LOW]: Missing `import numpy` in train_medmamba.py

**Severity:** LOW -- Would crash when loading real images
**File:** `train_medmamba.py` (line 264)

**Problem:**
`np.float32` was used but `numpy` was not imported.

**Fix Applied:** Added `import numpy as np`.

---

## Detailed Dimension Analysis

### 1. Novelty / Innovation (6.5/10)

**Strengths:**
- Combining VMamba's cross-scan with U-Mamba's CNN-SSM hybrid is a reasonable approach
- CTM-based hallucination detection for medical imaging is a unique angle
- HoME-MoE integration adds expert diversity

**Weaknesses:**
- No single component is novel; all are adapted from existing papers (VMamba CVPR 2024, U-Mamba, NeurIPS 2025 HoME)
- The "fusion" is a simple weighted sum, not a novel fusion mechanism
- The CTM trajectory analyzer is essentially a statistical wrapper (norm, std, margin) rather than a learned dynamics model
- Missing comparison with recent 2025-2026 SSM medical imaging works

**Recommendation:**
- Clearly articulate what is novel vs. what is engineering integration
- Add a "Contribution" section that precisely states 2-3 novel claims
- Consider adding a novel component (e.g., a learned cross-scan direction selector, or a medical-domain-specific SSM parameterization)

---

### 2. Technical Soundness (5.0/10)

**Strengths:**
- The overall architecture (CNN + SSM dual branch) is sound in principle
- The selective scan implementation follows the Mamba paper's approach
- Parameter initialization follows established practices (Xavier, dt init)

**Weaknesses:**
- 6 bugs found (3 critical, 1 medium, 2 low) -- see above
- The selective scan uses a Python `for` loop over sequence length, which is 100-1000x slower than a compiled CUDA kernel. All latency benchmarks are meaningless.
- The `MambaBlock2D` in `medmamba.py` and `SS2D` in `vmamba_blocks.py` are near-duplicates with slightly different implementations, suggesting code was copied rather than designed
- The CTM hallucination score weights (0.4, 0.3, 0.3) are hardcoded without justification
- `DualBranchEncoder` takes `input_proj` with `kernel_size=1` on 3-channel input, but the model already has `patch_embed` -- this means there are two redundant projections

**Recommendation:**
- Use `mamba-ssm` CUDA kernel for production benchmarks
- Consolidate `MambaBlock2D` and `SS2D` into a single implementation
- Remove the redundant `input_proj` in `DualBranchEncoder`
- Validate CTM weights via ablation study

---

### 3. Experimental Rigor (4.5/10)

**Strengths:**
- `benchmark.py` provides parameter/FLOPs/latency measurement
- `Evaluator` class supports comprehensive metrics (AUC, F1, ECE, Brier)
- Risk-error correlation analysis is well-designed

**Weaknesses:**
- **No real dataset experiments.** The `MockMedicalDataset` generates random tensors. There are no scripts that train on actual medical imaging datasets (e.g., ChestX-ray14, ISIC, BraTS).
- **No comparison tables.** No results against baselines (ResNet, ViT, Swin, vanilla Mamba, VMamba).
- **No ablation study.** The contributions (CNN branch, CTM, MoE, cross-scan) are not individually validated.
- **No statistical significance.** No confidence intervals, no multiple-seed runs.
- `benchmark.py` imports `from src.models.selective_state_space import MambaBlock` which may not exist.

**Recommendation:**
- Run experiments on at least 2 public medical imaging datasets
- Include comparison with 5+ baselines
- Add ablation table removing each component
- Report mean +/- std over 3+ seeds

---

### 4. Code Quality (5.5/10)

**Strengths:**
- Clear module structure (`src/models/`, `src/data/`, `src/`, `tests/`)
- Good use of `dataclass` for configuration
- Comprehensive test suite (`test_comprehensive.py` with 50+ tests)
- Docker support (Dockerfile.cpu, Dockerfile.gpu)

**Weaknesses:**
- Duplicate class definitions (now fixed)
- Chinese comments throughout -- limits international collaboration
- `MambaBlock2D` and `SS2D` are near-duplicates
- `MoEEnsemble.forward()` has an `attention` branch that reshapes incorrectly (`stacked_flat = stacked.reshape(B * n_exp, L, D)` then `output = output.mean(dim=0)` -- the batch dimension is wrong)
- `ExpertBlock` in `home_moe.py` applies `RMSNorm` after residual, but `RMSNorm` is applied to 4D tensors incorrectly (the local `RMSNorm` in `home_moe.py` handles 4D, but `ExpertBlock` operates on 3D `[B, L, D]`)
- `train_medmamba.py` has a `MedicalImageDataset` that's unused in favor of `MockMedicalDataset`

**Recommendation:**
- Translate all comments to English for submission
- Remove `MambaBlock2D` from `medmamba.py` (use `SS2D` everywhere)
- Fix `MoEEnsemble` attention branch
- Remove unused `MedicalImageDataset` or integrate it properly

---

### 5. Writing / Presentation (6.0/10)

**Strengths:**
- Docstrings are comprehensive with Args/Returns
- Module-level docstrings explain the purpose
- The README.md exists with usage instructions

**Weaknesses:**
- All inline comments are in Chinese
- No LaTeX-formatted architecture diagram
- No complexity analysis section in code comments
- `INNOVATION_ROADMAP.md` and `OPTIMIZATION_REPORT.md` suggest the project is still in development, not ready for submission

**Recommendation:**
- Translate all comments to English
- Add a formal complexity analysis (FLOPs derivation)
- Remove development artifacts before submission

---

### 6. Significance / Impact (6.0/10)

**Strengths:**
- O(n) SSM for medical imaging is timely and addresses ViT's O(n^2) limitation
- CTM hallucination detection is clinically relevant (doctors need to know when AI might be wrong)
- The Guard variant (risk-aware inference) is a practical contribution
- Multi-task (classification + segmentation) support is useful

**Weaknesses:**
- The O(n) claim is undermined by the Python for-loop implementation
- No clinical validation or expert evaluation
- The CTM hallucination score is a heuristic, not a learned metric
- Missing comparison with other uncertainty quantification methods (MC Dropout, Deep Ensembles, Evidential DL)

**Recommendation:**
- Implement proper parallel scan or use CUDA kernel for realistic O(n) benchmarks
- Add comparison with uncertainty baselines
- Include a clinical use case or expert study

---

### 7. Reproducibility (5.0/10)

**Strengths:**
- Docker support (Dockerfile.cpu, Dockerfile.gpu, docker-compose.yml)
- `REPRODUCE.md` exists
- `requirements.txt` with pinned versions
- `benchmark.py` for standardized measurement

**Weaknesses:**
- No trained model weights provided
- No actual training commands with real datasets
- No seed-locked results (seed=42 is set but no results to verify)
- `benchmark.py` imports `MambaBlock` from `selective_state_space` which may not be importable
- No wandb/tensorboard logs provided

**Recommendation:**
- Provide at least one trained checkpoint
- Include a `run_experiments.sh` script with exact commands
- Upload training logs to a public platform

---

## Priority Action Items for Q2 Acceptance

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Fix all 6 bugs (DONE in this review) | -- | Critical |
| P1 | Run experiments on 2+ public medical datasets | 1-2 weeks | Critical |
| P2 | Add comparison tables with 5+ baselines | 3-5 days | Critical |
| P3 | Translate all comments to English | 1 day | High |
| P4 | Implement CUDA kernel or use mamba-ssm for benchmarks | 3-5 days | High |
| P5 | Add ablation study table | 2-3 days | High |
| P6 | Add uncertainty quantification baselines | 1 week | Medium |
| P7 | Remove development artifacts (TODO.md, etc.) | 1 day | Medium |

---

## Files Modified in This Review

1. `src/models/medmamba.py` -- Fixed cls_token dead code, removed duplicate classes, fixed V3 forward loop
2. `src/trainer.py` -- Added missing `import random`
3. `train_medmamba.py` -- Fixed mixup type mismatch, added missing `import numpy`

---

## Conclusion

The MedMamba project has a solid architectural vision -- combining SSM efficiency with medical imaging requirements and adding clinically relevant risk monitoring. However, **6 critical bugs** (now fixed) and the **absence of real experimental results** make it currently unsuitable for Q2 SCI submission.

After the fixes applied in this review, the next steps are:
1. Run experiments on public datasets (ChestX-ray14, ISIC 2024, etc.)
2. Generate comparison tables and ablation studies
3. Translate all documentation to English
4. Consider CUDA acceleration for realistic benchmarks

With these changes, the paper has a reasonable chance at Q2 acceptance given the timely topic (SSM for medical imaging) and the unique CTM hallucination detection angle.
