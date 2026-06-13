# MedMamba-Guard API Reference

## REST API Endpoints

### Health Check

```
GET /health
```

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "model_loaded": true,
  "version": "MedMamba-Guard v1.0"
}
```

### Predict with Risk Assessment

```
POST /predict
```

Performs medical image classification with full risk assessment.

**Parameters:**
- `file` (multipart/form-data): Image file (jpg, png, dicom)

**Response:**
```json
{
  "success": true,
  "data": {
    "prediction": "Lesion",
    "confidence": 0.87,
    "risk_score": 0.42,
    "risk_components": {
      "R_state": 0.35,
      "R_scan": 0.28,
      "R_task": 0.15,
      "R_entropy": 0.22
    },
    "risk_level": "medium",
    "review_regions": [...],
    "audit_log": {...}
  }
}
```

### Risk-Only Assessment

```
POST /predict-risk-only
```

Returns only risk assessment without classification prediction.

### CTM Metrics

```
GET /ctm-metrics
```

Returns detailed CTM state trajectory metrics from the last inference.

---

## Python API

### Creating a Model

```python
from src.models.medmamba_guard import create_medmamba_guard

# Standard model
model = create_medmamba_guard(d_model=384, n_layers=12, num_classes=2)

# Lightweight model
from src.models.medmamba_guard import LightMedMambaGuard
model = LightMedMambaGuard(d_model=192, n_layers=6)

# Full model
from src.models.medmamba_guard import FullMedMambaGuard
model = FullMedMambaGuard(d_model=512, n_layers=16)
```

### Inference with Risk Assessment

```python
import torch
from src.models.medmamba_guard import create_medmamba_guard

model = create_medmamba_guard(d_model=384, n_layers=12, num_classes=2)
model.eval()

x = torch.randn(1, 3, 224, 224)

# Full prediction with risk
output = model.predict(x)
print(f"Prediction: {output['prediction']}")
print(f"Confidence: {output['confidence']:.2%}")
print(f"Risk Score: {output['risk_score']:.2f}")
print(f"Action: {output['action']}")
```

### Training

```python
from src.models.medmamba import create_medmamba
from src.trainer import Trainer, TrainerConfig

model = create_medmamba(version="v2", num_classes=2)
config = TrainerConfig(epochs=50, batch_size=16, learning_rate=1e-4)

trainer = Trainer(model=model, train_loader=train_loader, config=config)
history = trainer.train()
```

### Evaluation

```python
from src.evaluator import Evaluator, EvaluatorConfig

config = EvaluatorConfig(num_classes=2, compute_risk_metrics=True)
evaluator = Evaluator(model=model, data_loader=val_loader, config=config)
results = evaluator.run_evaluation()
```

---

## Risk Components

| Component | Description | Range |
|-----------|-------------|-------|
| R_state | CTM state trajectory risk | [0, 1] |
| R_scan | Cross-Scan consistency risk | [0, 1] |
| R_task | Classification-segmentation conflict | [0, 1] |
| R_entropy | Prediction entropy risk | [0, 1] |
| R_total | Weighted combination | [0, 1] |

## Gating Actions

| Action | Condition |
|--------|-----------|
| PASS | All risk thresholds within limits |
| doctor_review | R_total > 0.7 or R_task > 0.4 |
| overconfidence_warning | confidence > 0.85 and R_state > 0.5 |
