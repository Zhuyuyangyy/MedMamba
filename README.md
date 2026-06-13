<div align="center">

# MedMamba-Guard

**Trustworthy Medical Image Inference Framework Based on SSM State Trajectory Analysis**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-100%2B-brightgreen?style=flat-square)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-80%25+-blue?style=flat-square)](tests/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-yellow?style=flat-square&logo=githubactions)](.github/workflows/ci.yml)
[![Status](https://img.shields.io/badge/Status-Research%20Active-blue?style=flat-square)]()

</div>

---

## Overview

MedMamba-Guard is a trustworthy inference framework for medical imaging that augments State Space Model (SSM) based classifiers with multi-dimensional risk monitoring and hard gating mechanisms. Rather than treating the neural network as a black-box classifier, MedMamba-Guard exposes and analyzes the internal hidden state trajectories of the SSM to quantify prediction stability, detect spatial inconsistencies, and identify self-contradictory outputs before they reach the clinician.

The framework is built on a CNN-SSM dual-branch encoder architecture that combines the local spatial feature extraction of convolutional neural networks with the long-range sequence modeling of selective state spaces (VMamba-style four-direction scanning). On top of this backbone, four monitoring components operate in parallel: a CTM (Continuous Trajectory Monitor) that analyzes hidden state evolution across layers, a Cross-Scan Risk Analyzer that evaluates multi-directional feature consistency, a Task Conflict Validator that detects contradictions between classification and segmentation outputs, and a Hard Gating Rules engine that enforces clinician review when risk thresholds are exceeded.

MedMamba-Guard provides full audit logging of every inference, including per-layer state metrics, spatial risk heatmaps, and interpretable gating decisions. This design philosophy aligns with emerging regulatory requirements for AI-assisted diagnosis, where transparency, traceability, and human oversight are essential for clinical adoption.

---

## Key Features

- **CTM State Trajectory Monitoring** -- Quantifies SSM hidden state stability across layers using four metrics: state volatility (V_norm), input-dependent drift (D_delta), cross-layer semantic stability (C_layer), and overconfidence detection (R_overconf). Computes a unified R_state risk score from these components.

- **Cross-Scan Consistency Risk Analysis** -- Extends VMamba's four-direction scanning from feature enhancement to trustworthiness evaluation. Measures L2 and cosine divergence between directional features to detect spatially inconsistent high-risk regions.

- **Classification-Segmentation Mutual Verification** -- Detects internal model contradictions when classification confidence conflicts with spatial segmentation evidence, including area mismatch, compactness inconsistency, and boundary conflicts.

- **Hard Gating Rules Engine** -- Enforces three safety rules: (1) total risk exceeds threshold triggers doctor review, (2) task conflict exceeds threshold triggers doctor review, (3) high confidence with unstable internal state triggers overconfidence warning.

- **Audit Logging** -- Records model version, CTM metrics, Cross-Scan metrics, conflict metrics, and final gating decision for every inference, supporting full traceability for regulatory compliance.

- **Risk Heatmap Generation** -- Fuses CTM and Cross-Scan risk maps into a unified spatial heatmap that highlights regions requiring clinician attention.

- **Modular Architecture** -- Each monitoring component (CTM, Cross-Scan, Task Validator, Hard Gating) can be independently enabled or disabled for ablation studies and deployment flexibility.

- **REST API with FastAPI** -- Serves predictions with risk assessment via HTTP endpoints, supporting both full prediction and risk-only evaluation modes.

---

## Architecture

```
Input Image [B, 3, H, W]
       |
       v
+---------------------------+
|   Input Projection (1x1)  |
|   [B, d_model, H, W]     |
+---------------------------+
       |
       |  x N layers
       v
+-------------------------------------------+
|   CNN-SSM Dual-Branch Encoder              |
|                                            |
|   SSM Branch: VSSBlock2D (4-dir scan)      |
|   CNN Branch: DWConv + PWConv residual     |
|   Feature Fusion: learnable weight blend   |
+-------------------------------------------+
       |
       |  Hidden states {h_1, h_2, ..., h_t}
       v
+----------------+  +------------------+  +------------------+
| CTM State      |  | Cross-Scan       |  | Task Conflict    |
| Trajectory     |  | Consistency      |  | Validator        |
| Monitor        |  | Risk Analyzer    |  |                  |
|                |  |                  |  | P_cls vs M_seg   |
| V_norm, D_dlt  |  | L2 div, Cos div  |  | Area/Compact/    |
| C_layer, R_ocf |  | R_scan score     |  | Boundary conflict|
+-------+--------+  +--------+---------+  +--------+---------+
        |                    |                      |
        +--------------------+----------------------+
                             |
                             v
                  +---------------------+
                  |  Hard Gating Rules  |
                  |                     |
                  |  R_total = w_s*R_s  |
                  |    + w_sc*R_sc      |
                  |    + w_t*R_t        |
                  |    + w_e*R_e        |
                  |                     |
                  |  Action:            |
                  |   PASS /            |
                  |   doctor_review /   |
                  |   overconfidence_   |
                  |     warning         |
                  +----------+----------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +------------+  +-------------+  +------------+
     | Prediction |  | Risk Score  |  | Audit Log  |
     | + Confidence|  | + Heatmap  |  | + Gating   |
     +------------+  +-------------+  +------------+
```

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Deep Learning Framework | PyTorch | 2.0+ |
| Sequence Modeling | Selective State Space (VMamba) | -- |
| Tensor Operations | einops | 0.7+ |
| API Framework | FastAPI | 0.100+ |
| ASGI Server | Uvicorn | 0.23+ |
| Data Processing | NumPy, Pillow | >=1.21, >=9.0 |
| Containerization | Docker (CPU + GPU) | 24+ |
| Orchestration | Docker Compose | -- |
| CI/CD | GitHub Actions | -- |

---

## Quick Start

### Prerequisites

- Python 3.9 or later
- PyTorch 2.0+
- CUDA 11.7+ (for GPU inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/MedMamba.git
cd MedMamba

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.models.medmamba import MedMambaV2; print('Installation successful')"
```

### Docker

```bash
# CPU version
docker build -f Dockerfile.cpu -t medmamba:cpu .
docker run -p 8866:8866 medmamba:cpu

# GPU version
docker build -f Dockerfile.gpu -t medmamba:gpu .
docker run --gpus all -p 8866:8866 medmamba:gpu

# Docker Compose
docker-compose up -d
```

### Interactive Menu

```bash
bash start.sh

# Options:
#   1 - Info:         View model architecture information
#   2 - Benchmark:    Test model performance
#   3 - Train:        Train the model
#   4 - Serve:        Start API service (port 8866)
#   5 - Guard-Demo:   Risk assessment demo (no training required)
#   6 - Risk-Analysis: Risk-error correlation analysis
#   7 - Ablation:     Ablation experiments
```

### Python API

```python
from src.models.medmamba_guard import create_medmamba_guard
import torch

# Create model
model = create_medmamba_guard(d_model=384, n_layers=12, num_classes=2)

# Inference with full risk assessment
x = torch.randn(1, 3, 224, 224)
output = model.predict(x)

print(f"Prediction: {output['prediction']}")
print(f"Confidence: {output['confidence']:.2%}")
print(f"Risk Score: {output['risk_score']:.2f}")
print(f"Action: {output['action']}")
```

### REST API

```bash
# Start server
python main.py

# Predict with risk assessment
curl -X POST http://localhost:8866/predict \
  -F "file=@test_lesion.jpg"

# Risk-only assessment
curl -X POST http://localhost:8866/predict-risk-only \
  -F "file=@test_lesion.jpg"

# CTM metrics
curl http://localhost:8866/ctm-metrics

# Health check
curl http://localhost:8866/health
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test suite
pytest tests/test_comprehensive.py -v
```

### CI/CD

The project uses GitHub Actions for continuous integration:

- **Lint**: Ruff code formatting and style checks
- **Test**: Pytest across Python 3.9-3.12 with 80%+ coverage requirement
- **Security**: Dependency vulnerability scanning with `safety` and `bandit`
- **Docker**: Automated image build and smoke test on main branch

---

## Project Structure

```
MedMamba/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py              # FastAPI service
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py        # Data augmentation pipelines
│   │   ├── dataset.py             # Medical image dataset
│   │   └── dataset_guard.py       # Guard-specific dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── medmamba.py            # MedMamba core (V2/V3)
│   │   ├── medmamba_guard.py      # Guard framework (main model)
│   │   ├── vmamba_blocks.py       # VMamba four-direction scanning
│   │   ├── home_moe.py            # HoME-MoE expert mixture
│   │   ├── selective_state_space.py # SSM core implementation
│   │   ├── ssm_config.py          # Configuration definitions
│   │   ├── ctm_monitor.py         # CTM state monitoring
│   │   ├── cross_scan_risk.py     # Cross-Scan risk analysis
│   │   └── task_conflict_validator.py # Task conflict validation
│   ├── evaluator.py               # Evaluation utilities
│   └── trainer.py                 # Training utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared test fixtures
│   ├── test_smoke.py              # Smoke tests (~40 cases)
│   ├── test_guard_core.py         # Guard core tests (~20 cases)
│   └── test_comprehensive.py      # Comprehensive tests (100+ cases)
├── experiments/
│   ├── ablation_guard.py          # Guard ablation experiments
│   ├── analyze_smoke.py           # Smoke test analysis
│   └── risk_error_analysis.py     # Risk-error correlation analysis
├── scripts/
│   └── run_ablation_study.py      # Ablation experiment runner
├── docs/
│   ├── API_REFERENCE.md           # REST & Python API documentation
│   ├── ARCHITECTURE.md            # System architecture guide
│   ├── DEPLOYMENT.md              # Deployment guide
│   ├── MEDMAMBA_GUARD_TECHNICAL_REPORT.md # Technical report
│   ├── paper_framework.md         # Paper framework
│   └── figure_guide.md            # Visualization guide
├── frontend/
│   └── index.html                 # Web demo interface
├── weights/                       # Model weights directory
├── main.py                        # Main entry point
├── train_medmamba.py              # Standard training script
├── train_medmamba_guard.py        # Guard training script
├── benchmark.py                   # Performance benchmark
├── start.sh                       # Launch script
├── requirements.txt               # Dependencies
├── Dockerfile.cpu                 # CPU Docker image
├── Dockerfile.gpu                 # GPU Docker image
├── docker-compose.yml             # Docker orchestration
├── .github/workflows/ci.yml       # CI/CD configuration
├── TODO.md                        # Innovation suggestions & technical debt
├── INNOVATION_ROADMAP.md          # Patent proposals & research roadmap
├── OPTIMIZATION_REPORT.md         # Project optimization report
├── REPRODUCE.md                   # Reproduction guide
├── LICENSE                        # MIT License
└── .gitignore
```

---

## Benchmarks & Results

### Model Complexity

| Model | Parameters | FLOPs | Inference Time |
|-------|-----------|-------|---------------|
| MedMamba-V2 (d=384) | ~45M | ~8G | ~15ms/image |
| MedMamba-V3 (d=384) | ~52M | ~10G | ~18ms/image |
| MedMamba-Guard | ~48M | ~9G | ~20ms/image |
| Light-Guard (d=192) | ~12M | ~2G | ~8ms/image |
| ViT-B/16 (reference) | ~86M | ~17G | ~15ms/image |

### Risk Assessment Performance

| Metric | Description | Target | V1 | V2 |
|--------|------------|--------|-----|-----|
| AUROC_error | Risk score AUC for predicting model errors | >= 0.80 | 0.82 | 0.85 |
| ECR@10% | Error capture rate in top-10% high-risk samples | >= 55% | 58% | 62% |
| Accuracy Loss | Accuracy degradation vs. unguarded model | < 1% | 0.3% | 0.2% |

### Evaluation Metrics

| Metric | Description |
|--------|------------|
| Accuracy / Dice | Classification accuracy and segmentation overlap |
| AUROC_error | AUC of risk score for predicting model errors |
| ECR@10% | Proportion of errors captured in top-10% risk samples |
| R_total | Composite risk score (state + scan + task + entropy) |
| R_state | CTM state trajectory risk component |
| R_scan | Cross-Scan consistency risk component |

### Ablation Experiments

| Experiment | Disabled Component | Verified Innovation |
|------------|-------------------|-------------------|
| w/o CTM | CTMMonitor | CTM state trajectory monitoring |
| w/o CrossScan | CrossScanRiskAnalyzer | Cross-Scan consistency |
| w/o clDice | TaskConflictValidator | Classification-segmentation mutual verification |
| w/o SafeMamba | HardGatingRules | Doctor review gating |
| Baseline | No Guard | Standard MedMamba without Guard |

---

## Research & Publications

### Citation

```bibtex
@article{medmamba2025guard,
  title={MedMamba-Guard: A Trustworthy Inference Framework for Medical Imaging Based on SSM State Trajectory Analysis},
  author={Chen, L. and Wang, X. and Liu, Y. and Zhang, J.},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

### Related Work

- **Mamba** -- Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2023.
- **VMamba** -- Liu, Y. et al. "VMamba: Visual State Space Model." arXiv 2024.
- **MedMamba** -- Original medical image classification with SSM architecture.

---

## Innovation & Patents

The project includes 4 patent-ready innovations. See [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md) for details.

1. **CTM State Trajectory Monitoring** -- Real-time hallucination detection using SSM hidden state dynamics
2. **Cross-Scan Consistency Risk Analysis** -- Multi-directional feature verification for VMamba outputs
3. **Classification-Segmentation Mutual Verification** -- Dual-task consistency gating
4. **Hierarchical MoE for Medical Imaging** -- Multi-scale expert routing with trustworthiness monitoring

---

## Roadmap

- [x] V1: Base MedMamba + CTM monitoring
- [x] V2: VMamba four-direction scanning + CNN-SSM fusion
- [x] V3: HoME-MoE expert mixture routing
- [ ] V4: Multi-modal support (CT + MRI + pathology)
- [ ] V5: 3D medical image segmentation
- [ ] V6: Few-shot learning for rare diseases
- [ ] V7: Federated learning integration
- [ ] V8: Regulatory pathway documentation (NMPA/FDA)

See [TODO.md](TODO.md) for detailed innovation suggestions and technical debt.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- Issues and bug reports: [GitHub Issues](https://github.com/your-org/MedMamba/issues)
- Technical documentation: `docs/MEDMAMBA_GUARD_TECHNICAL_REPORT.md`
- Innovation details: `docs/INNOVATION.md`

---

<div align="center">

**MedMamba-Guard** -- From black-box classifier to trustworthy clinical decision support.

</div>
