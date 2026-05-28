# REPRODUCE.md - MedMamba

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Recommended for training; CPU for inference/demo
- **PyTorch**: 2.0+

## Install

```bash
cd MedMamba
pip install -r requirements.txt
```

Core dependencies: torch, einops, numpy, Pillow, fastapi, uvicorn

Optional (for acceleration):
```bash
# pip install mamba-ssm>=1.2.0  # CUDA SSM
# pip install flash-attn>=2.0.0  # FlashAttention
```

## Smoke Test

```bash
python -m pytest tests/ -v
```

Expected: `tests/test_guard_core.py` passes.

## Run Main

```bash
python main.py
```

## Run Benchmark

```bash
python benchmark.py
```

## Expected Outputs

- Smoke test results: `docs/demo_evidence_v0.2_lite/smoke_results.csv` and `.json`
- CTM state trajectory monitoring metrics
- Cross-scan consistency risk maps

## Known Issues

- `mamba-ssm` and `flash-attn` are optional but significantly speed up SSM operations
- No training dataset included; requires medical image datasets
- Core demo works without GPU using synthetic data

## Architecture

Four innovations:
1. CTM state trajectory monitoring (V_norm, D_Δ, C_layer, R_overconf)
2. Cross-scan consistency risk maps
3. Guard intervention mechanism
4. Evidence chain tracking
