# MedMamba-Guard Architecture

## System Overview

MedMamba-Guard is a trustworthy inference framework for medical imaging that augments State Space Model (SSM) based classifiers with multi-dimensional risk monitoring and hard gating mechanisms.

## Core Architecture

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
                  +----------+----------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
     +------------+  +-------------+  +------------+
     | Prediction |  | Risk Score  |  | Audit Log  |
     +------------+  +-------------+  +------------+
```

## Key Components

### 1. CNN-SSM Dual-Branch Encoder

Combines local spatial feature extraction (CNN) with long-range sequence modeling (SSM):

- **SSM Branch**: VSSBlock2D with four-direction scanning (VMamba-style)
- **CNN Branch**: Depthwise separable convolution with residual connections
- **Feature Fusion**: Learnable weighted combination

### 2. CTM State Trajectory Monitor

Analyzes SSM hidden state evolution across layers:

- **V_norm**: State transition instability
- **D_delta**: Input-dependent response drift
- **C_layer**: Cross-layer semantic stability
- **R_overconf**: Overconfidence detection

### 3. Cross-Scan Consistency Risk Analyzer

Extends VMamba's four-direction scanning for trustworthiness:

- Computes L2 and cosine divergence between directional features
- Detects spatially inconsistent high-risk regions

### 4. Task Conflict Validator

Detects contradictions between classification and segmentation:

- Area mismatch detection
- Compactness inconsistency
- Boundary stability analysis

### 5. Hard Gating Rules Engine

Enforces safety rules:

- R_total > threshold: doctor_review
- R_task > threshold: doctor_review
- High confidence + unstable state: overconfidence_warning

## Model Variants

| Variant | d_model | n_layers | Parameters | Use Case |
|---------|---------|----------|------------|----------|
| Light | 192 | 6 | ~12M | Edge deployment |
| Standard | 384 | 12 | ~48M | Clinical inference |
| Full | 512 | 16 | ~85M | Research/benchmark |
