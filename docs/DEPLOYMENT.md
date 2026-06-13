# Deployment Guide

## Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/your-org/MedMamba.git
cd MedMamba
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start API server
python src/api/server.py
```

### Docker Deployment

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

## Production Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DEVICE | auto | Device (cpu/cuda) |
| IMG_SIZE | 224 | Input image size |
| PORT | 8866 | API port |
| MODEL_PATH | - | Path to model weights |

### Health Check

```bash
curl http://localhost:8866/health
```

### Scaling

For production workloads:

1. Use GPU-enabled Docker image
2. Deploy behind a load balancer
3. Use Kubernetes for orchestration
4. Enable model caching

## CI/CD Pipeline

The project includes GitHub Actions CI/CD:

1. **Lint**: Ruff code formatting and style checks
2. **Test**: Pytest with coverage reporting (80%+ target)
3. **Security**: Dependency vulnerability scanning
4. **Docker**: Automated image building on main branch

## Monitoring

- Health endpoint: `/health`
- CTM metrics: `/ctm-metrics`
- Audit logs: Included in every prediction response
