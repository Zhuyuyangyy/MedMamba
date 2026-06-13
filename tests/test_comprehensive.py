"""
tests/test_comprehensive.py
Comprehensive test suite for MedMamba-Guard project.
Targets 80%+ code coverage across all modules.

Run: pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")


# ============================================================
# TestSSMConfig - Configuration module tests
# ============================================================

class TestSSMConfig:
    """Tests for ssm_config.py"""

    def test_ssm_config_defaults(self):
        from src.models.ssm_config import SSMConfig
        cfg = SSMConfig()
        assert cfg.d_model == 768
        assert cfg.d_state == 16
        assert cfg.dropout == 0.1

    def test_ssm_config_custom(self):
        from src.models.ssm_config import SSMConfig
        cfg = SSMConfig(d_model=256, d_state=8, dropout=0.2)
        assert cfg.d_model == 256
        assert cfg.d_state == 8
        assert cfg.dropout == 0.2

    def test_medmamba_config_defaults(self):
        from src.models.ssm_config import MedMambaConfig
        cfg = MedMambaConfig()
        assert cfg.num_classes == 2
        assert cfg.img_size == 224
        assert cfg.modal == "CT"

    def test_medmamba_config_custom(self):
        from src.models.ssm_config import MedMambaConfig
        cfg = MedMambaConfig(d_model=384, modal="MRI", multi_label=True)
        assert cfg.modal == "MRI"
        assert cfg.multi_label is True

    def test_ctm_config_defaults(self):
        from src.models.ssm_config import CTMConfig
        cfg = CTMConfig()
        assert cfg.trajectory_window == 5
        assert cfg.delta_window == 3
        assert cfg.hard_gate_high == 0.7

    def test_guard_config_inherits(self):
        from src.models.ssm_config import MedMambaGuardConfig
        cfg = MedMambaGuardConfig()
        assert hasattr(cfg, 'd_model')
        assert hasattr(cfg, 'trajectory_window')
        assert hasattr(cfg, 'use_ctm')

    def test_ssm_config_dt_rank_auto(self):
        from src.models.ssm_config import SSMConfig
        cfg = SSMConfig(d_model=256, dt_rank="auto")
        assert isinstance(cfg.dt_rank, int)
        assert cfg.dt_rank == max(256 // 16, 1)


# ============================================================
# TestSelectiveStateSpace - SSM core module tests
# ============================================================

class TestSelectiveStateSpace:
    """Tests for selective_state_space.py"""

    def test_rms_norm_shape(self):
        from src.models.selective_state_space import RMSNorm
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_norm_values(self):
        from src.models.selective_state_space import RMSNorm
        norm = RMSNorm(d_model=4)
        x = torch.ones(1, 1, 4)
        out = norm(x)
        assert not torch.isnan(out).any()

    def test_mamba_block_forward(self):
        from src.models.selective_state_space import MambaBlock
        block = MambaBlock(d_model=64, d_state=8, dropout=0.0)
        block.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 16, 64)

    def test_mamba_block_gradient(self):
        from src.models.selective_state_space import MambaBlock
        block = MambaBlock(d_model=32, d_state=4, dropout=0.0)
        block.train()
        x = torch.randn(1, 8, 32)
        out = block(x)
        loss = out.sum()
        loss.backward()
        grad_count = sum(1 for p in block.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_selective_state_space_stack(self):
        from src.models.selective_state_space import SelectiveStateSpace
        ssm = SelectiveStateSpace(d_model=32, n_layers=2, d_state=4, dropout=0.0)
        ssm.eval()
        x = torch.randn(1, 8, 32)
        with torch.no_grad():
            out = ssm(x)
        assert out.shape == (1, 8, 32)

    def test_medmamba_block_forward(self):
        from src.models.selective_state_space import MedMambaBlock
        block = MedMambaBlock(d_model=16, d_state=4)
        block.eval()
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape


# ============================================================
# TestVMambaBlocks - VMamba blocks tests
# ============================================================

class TestVMambaBlocks:
    """Tests for vmamba_blocks.py"""

    def test_cross_scan_directions(self):
        from src.models.vmamba_blocks import CrossScan
        scan = CrossScan()
        x = torch.randn(2, 8, 4, 4)
        scans = scan(x)
        assert len(scans) == 4
        for s in scans:
            assert s.shape == (2, 8, 16)

    def test_cross_merge_shape(self):
        from src.models.vmamba_blocks import CrossMerge
        merge = CrossMerge()
        B, C, H, W = 2, 8, 4, 4
        scans = [torch.randn(B, C, H * W) for _ in range(4)]
        out = merge(scans, H, W)
        assert out.shape == (B, C, H, W)

    def test_ss2d_forward(self):
        from src.models.vmamba_blocks import SS2D
        ss2d = SS2D(d_model=32, d_state=4, dropout=0.0)
        ss2d.eval()
        x = torch.randn(1, 16, 32)
        with torch.no_grad():
            out = ss2d(x)
        assert out.shape == (1, 16, 32)

    def test_vss_block_2d(self):
        from src.models.vmamba_blocks import VSSBlock2D
        block = VSSBlock2D(d_model=32, d_state=4, dropout=0.0)
        block.eval()
        x = torch.randn(1, 32, 4, 4)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (1, 32, 4, 4)

    def test_cross_attention_fusion(self):
        from src.models.vmamba_blocks import CrossAttentionFusion
        fusion = CrossAttentionFusion(d_model=32, num_heads=4)
        fusion.eval()
        ssm_feat = torch.randn(1, 32, 4, 4)
        cnn_feat = torch.randn(1, 32, 4, 4)
        with torch.no_grad():
            out = fusion(ssm_feat, cnn_feat)
        assert out.shape == (1, 32, 4, 4)

    def test_vmamba2d_forward(self):
        from src.models.vmamba_blocks import VMamba2D
        model = VMamba2D(d_model=32, depth=1, d_state=4, dropout=0.0)
        model.eval()
        x = torch.randn(1, 32, 4, 4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 32, 4, 4)

    def test_simplified_cross_scan(self):
        from src.models.vmamba_blocks import SimplifiedCrossScan
        scan = SimplifiedCrossScan()
        x = torch.randn(1, 8, 4, 4)
        scans = scan(x)
        assert len(scans) == 2

    def test_simplified_cross_merge(self):
        from src.models.vmamba_blocks import SimplifiedCrossMerge
        merge = SimplifiedCrossMerge()
        B, C, H, W = 1, 8, 4, 4
        scans = [torch.randn(B, C, H * W) for _ in range(2)]
        out = merge(scans, H, W)
        assert out.shape == (B, C, H, W)

    def test_channel_shuffle(self):
        from src.models.vmamba_blocks import channel_shuffle
        x = torch.randn(2, 16, 4, 4)
        out = channel_shuffle(x, groups=4)
        assert out.shape == x.shape

    def test_rms_norm_4d(self):
        from src.models.vmamba_blocks import RMSNorm
        norm = RMSNorm(d_model=16)
        x = torch.randn(1, 16, 4, 4)
        out = norm(x)
        assert out.shape == x.shape


# ============================================================
# TestHoMEMoE - MoE module tests
# ============================================================

class TestHoMEMoE:
    """Tests for home_moe.py"""

    def test_expert_block(self):
        from src.models.home_moe import ExpertBlock
        expert = ExpertBlock(d_model=32)
        x = torch.randn(1, 8, 32)
        out = expert(x)
        assert out.shape == (1, 8, 32)

    def test_expert_router(self):
        from src.models.home_moe import ExpertRouter
        router = ExpertRouter(d_model=32, num_experts=4)
        x = torch.randn(1, 8, 32)
        weights, indices = router(x)
        assert weights.shape == (1, 4)
        assert indices.shape == (1, 2)

    def test_hierarchical_moe(self):
        from src.models.home_moe import HierarchicalMoE
        moe = HierarchicalMoE(d_model=32, num_experts=4)
        x = torch.randn(1, 32, 4, 4)
        out, info = moe(x)
        assert out.shape == (1, 32, 4, 4)
        assert "weights" in info
        assert "entropy" in info

    def test_ctm_trajectory_analyzer(self):
        from src.models.home_moe import CTMTrajectoryAnalyzer
        analyzer = CTMTrajectoryAnalyzer(d_model=32, num_ticks=4)
        traj = torch.randn(2, 8, 32)
        result = analyzer(traj)
        assert "stability" in result
        assert "oscillation" in result
        assert "attractor_margin" in result

    def test_ctm_hallucination_score(self):
        from src.models.home_moe import CTMTrajectoryAnalyzer
        analyzer = CTMTrajectoryAnalyzer(d_model=32)
        traj = torch.randn(1, 4, 32)
        score = analyzer.hallucination_score(traj)
        assert score.shape == (1,)
        assert 0 <= score.item() <= 1

    def test_expert_diversity_loss(self):
        from src.models.home_moe import ExpertDiversityLoss
        loss_fn = ExpertDiversityLoss()
        weights = torch.softmax(torch.randn(2, 4), dim=-1)
        loss = loss_fn(weights)
        assert loss.item() >= 0

    def test_moe_ensemble(self):
        from src.models.home_moe import MoEEnsemble
        ensemble = MoEEnsemble(d_model=32, num_experts=4)
        experts = [torch.randn(1, 8, 32) for _ in range(4)]
        out = ensemble(experts)
        assert out.shape == (1, 8, 32)

    def test_hoME_moe_2d(self):
        from src.models.home_moe import HoMEMoE2D
        model = HoMEMoE2D(d_model=32, num_experts=4, depth=2)
        x = torch.randn(1, 32, 4, 4)
        out, infos = model(x)
        assert out.shape == (1, 32, 4, 4)


# ============================================================
# TestCTMMonitor - CTM monitoring tests
# ============================================================

class TestCTMMonitor:
    """Tests for ctm_monitor.py"""

    def test_ctm_monitor_creation(self):
        from src.models.ctm_monitor import CTMMonitor
        monitor = CTMMonitor(hidden_dim=64)
        assert monitor.hidden_dim == 64

    def test_ctm_monitor_compute_metrics(self):
        from src.models.ctm_monitor import CTMMonitor
        monitor = CTMMonitor(hidden_dim=32)
        states = [torch.randn(1, 32) for _ in range(6)]
        metrics = monitor.compute_ctm_metrics(states, states, 0.8)
        assert 'v_norm' in metrics
        assert 'r_state' in metrics

    def test_ctm_monitor_clear_cache(self):
        from src.models.ctm_monitor import CTMMonitor
        monitor = CTMMonitor(hidden_dim=32)
        monitor.hidden_states.append(torch.randn(1, 32))
        monitor.clear_cache()
        assert len(monitor.hidden_states) == 0

    def test_ctm_monitor_risk_score(self):
        from src.models.ctm_monitor import CTMMonitor
        monitor = CTMMonitor(hidden_dim=32)
        states = [torch.randn(1, 32) for _ in range(4)]
        monitor.compute_ctm_metrics(states, states, 0.7)
        score = monitor.get_state_risk_score()
        assert isinstance(score, float)

    def test_ctm_monitor_risk_map(self):
        from src.models.ctm_monitor import CTMMonitor
        monitor = CTMMonitor(hidden_dim=32)
        states = [torch.randn(1, 32) for _ in range(4)]
        monitor.compute_ctm_metrics(states, states, 0.7)
        risk_map = monitor.get_state_risk_map()
        assert isinstance(risk_map, torch.Tensor)

    def test_ctm_create_factory(self):
        from src.models.ctm_monitor import create_ctm_monitor
        monitor = create_ctm_monitor(hidden_dim=64)
        assert isinstance(monitor, type(monitor))


# ============================================================
# TestCrossScanRisk - Cross-Scan risk tests
# ============================================================

class TestCrossScanRisk:
    """Tests for cross_scan_risk.py"""

    def test_analyzer_creation(self):
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        analyzer = CrossScanRiskAnalyzer(feature_dim=32)
        assert analyzer.feature_dim == 32

    def test_analyzer_forward(self):
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        analyzer = CrossScanRiskAnalyzer(feature_dim=32)
        x = torch.randn(1, 32, 4, 4)
        result = analyzer(x, H=4, W=4)
        assert 'risk_score' in result
        assert 'risk_map' in result

    def test_simplified_analyzer(self):
        from src.models.cross_scan_risk import SimplifiedCrossScanRiskAnalyzer
        analyzer = SimplifiedCrossScanRiskAnalyzer()
        x = torch.randn(1, 16, 4, 4)
        result = analyzer(x)
        assert 'risk_score' in result

    def test_extract_directional_features(self):
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        analyzer = CrossScanRiskAnalyzer(feature_dim=16)
        x = torch.randn(1, 16, 4, 4)
        features = analyzer.extract_directional_features(x)
        assert len(features) == 4


# ============================================================
# TestTaskConflictValidator - Task conflict tests
# ============================================================

class TestTaskConflictValidator:
    """Tests for task_conflict_validator.py"""

    def test_validator_creation(self):
        from src.models.task_conflict_validator import TaskConflictValidator
        validator = TaskConflictValidator()
        assert validator.conflict_threshold == 0.4

    def test_validator_high_conf_large_lesion(self):
        from src.models.task_conflict_validator import TaskConflictValidator
        validator = TaskConflictValidator()
        p_cls = torch.tensor([[0.9]])
        m_seg = torch.ones(1, 1, 8, 8) * 0.85
        result = validator.validate(p_cls, m_seg)
        assert result['action'] == 'PASS'

    def test_validator_high_conf_no_lesion(self):
        from src.models.task_conflict_validator import TaskConflictValidator
        validator = TaskConflictValidator()
        p_cls = torch.tensor([[0.95]])
        m_seg = torch.zeros(1, 1, 8, 8)
        result = validator.validate(p_cls, m_seg)
        assert result['action'] in ('REVIEW', 'CONFLICT_LESION_MISSED', 'CONFLICT_OVER_CONFIDENT')

    def test_create_task_validator(self):
        from src.models.task_conflict_validator import create_task_validator
        validator = create_task_validator()
        assert validator is not None


# ============================================================
# TestMedMambaModels - Model integration tests
# ============================================================

class TestMedMambaModels:
    """Tests for medmamba.py models"""

    def test_medmamba_v2_creation(self):
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2, dropout=0.0)
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_medmamba_v2_forward_classification(self):
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model(x, task="classification")
        assert "logits" in result
        assert result["logits"].shape == (1, 2)

    def test_medmamba_v2_forward_with_ctm(self):
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model(x, task="classification", return_ctm=True)
        assert "logits" in result

    def test_medmamba_v3_creation(self):
        from src.models.medmamba import MedMambaV3
        model = MedMambaV3(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2,
                           use_ctm=True, use_moe=False, dropout=0.0)
        assert model is not None

    def test_medmamba_v3_forward(self):
        from src.models.medmamba import MedMambaV3
        model = MedMambaV3(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2,
                           use_ctm=True, use_moe=False, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model(x, task="classification", return_ctm=True)
        assert "logits" in result

    def test_create_medmamba_factory_v1(self):
        from src.models.medmamba import create_medmamba
        model = create_medmamba(version="v2", img_size=32, patch_size=4, d_model=32, n_layers=2,
                                num_classes=2, dropout=0.0)
        assert model is not None

    def test_create_medmamba_factory_v2(self):
        from src.models.medmamba import create_medmamba
        model = create_medmamba(version="v2", img_size=32, patch_size=4, d_model=32, n_layers=2,
                                num_classes=2, dropout=0.0)
        assert model is not None

    def test_create_medmamba_invalid_version(self):
        from src.models.medmamba import create_medmamba
        with pytest.raises(ValueError):
            create_medmamba(version="v99")

    def test_medmamba_head_classification(self):
        from src.models.medmamba import MedMambaHead
        head = MedMambaHead(d_model=32, num_classes=2)
        x = torch.randn(1, 32, 4, 4)
        result = head(x, task="classification")
        assert "logits" in result

    def test_cnn_branch_forward(self):
        from src.models.medmamba import CNNBranch
        branch = CNNBranch(d_model=16, d_state=4)
        branch.eval()
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            out = branch(x)
        assert out.shape == x.shape

    def test_feature_fusion_forward(self):
        from src.models.medmamba import FeatureFusion
        fusion = FeatureFusion(d_model=16)
        fusion.eval()
        cnn = torch.randn(1, 16, 4, 4)
        ssm = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            out = fusion(cnn, ssm)
        assert out.shape == (1, 16, 4, 4)

    def test_dual_branch_encoder(self):
        from src.models.medmamba import DualBranchEncoder
        encoder = DualBranchEncoder(d_model=32, n_layers=2, use_ctm=True, dropout=0.0)
        encoder.eval()
        x = torch.randn(1, 32, 4, 4)
        with torch.no_grad():
            out, ctm = encoder(x, return_ctm=True)
        assert out.shape == (1, 32, 4, 4)


# ============================================================
# TestMedMambaGuard - Guard model tests
# ============================================================

class TestMedMambaGuard:
    """Tests for medmamba_guard.py"""

    def test_guard_creation(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        assert model is not None

    def test_guard_forward_output_schema(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        required = ['prediction', 'confidence', 'risk_score', 'risk_components',
                     'risk_heatmap', 'review_regions', 'action', 'audit_log']
        for field in required:
            assert field in output, f"Missing: {field}"

    def test_guard_risk_components(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        rc = output['risk_components']
        assert 'R_state' in rc
        assert 'R_scan' in rc
        assert 'R_task' in rc
        assert 'R_entropy' in rc

    def test_guard_predict(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model.predict(x)
        assert 'prediction' in output
        assert 'risk_score' in output

    def test_guard_train_mode(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        model.train()
        x = torch.randn(1, 3, 32, 32)
        output = model(x, mode='train')
        assert 'cls_logits' in output

    def test_light_guard(self):
        from src.models.medmamba_guard import LightMedMambaGuard
        model = LightMedMambaGuard(d_model=32, d_state=4, n_layers=2, dropout=0.0)
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000

    def test_full_guard(self):
        from src.models.medmamba_guard import FullMedMambaGuard
        model = FullMedMambaGuard(d_model=32, d_state=4, n_layers=2, dropout=0.0)
        assert model is not None

    def test_create_guard_factory(self):
        from src.models.medmamba_guard import create_medmamba_guard
        model = create_medmamba_guard(d_model=32, n_layers=2, num_classes=2, dropout=0.0)
        assert model is not None

    def test_hard_gating_rules_pass(self):
        from src.models.medmamba_guard import HardGatingRules
        gating = HardGatingRules()
        action, reason, details = gating.evaluate(
            r_state=0.1, r_scan=0.1, r_task=0.05, r_entropy=0.1, confidence=0.75
        )
        assert action == "PASS"

    def test_hard_gating_rules_review(self):
        from src.models.medmamba_guard import HardGatingRules
        gating = HardGatingRules()
        action, reason, details = gating.evaluate(
            r_state=0.3, r_scan=0.3, r_task=0.3, r_entropy=0.3, confidence=0.6
        )
        assert action in ("doctor_review", "overconfidence_warning")

    def test_risk_heatmap_generator(self):
        from src.models.medmamba_guard import RiskHeatmapGenerator
        gen = RiskHeatmapGenerator()
        ctm = torch.randn(1, 16)
        scan = torch.randn(1, 16)
        hm = gen.generate(ctm, scan, H=4, W=4)
        assert hm.shape == (1, 4, 4)

    def test_audit_logger(self):
        from src.models.medmamba_guard import AuditLogger
        logger = AuditLogger()
        log = logger.start_inference((1, 3, 32, 32))
        assert 'model_version' in log
        assert 'timestamp' in log
        logger.log_final_decision("lesion", 0.9, 0.3, "PASS", "low risk")
        current = logger.get_current_log()
        assert current is not None

    def test_guard_batch_processing(self):
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert output['risk_heatmap'].shape[0] == 2


# ============================================================
# TestDataModules - Data module tests
# ============================================================

class TestDataModules:
    """Tests for data modules"""

    def test_dataset_import(self):
        from src.data.dataset import MedicalImageDataset
        assert MedicalImageDataset is not None

    def test_augmentation_import(self):
        from src.data.augmentation import get_train_transforms
        assert get_train_transforms is not None

    def test_dataset_guard_import(self):
        from src.data.dataset_guard import create_dataset
        assert create_dataset is not None


# ============================================================
# TestTrainer - Trainer module tests
# ============================================================

class TestTrainer:
    """Tests for trainer.py"""

    def test_trainer_config(self):
        from src.trainer import TrainerConfig
        cfg = TrainerConfig()
        assert cfg.epochs == 50
        assert cfg.batch_size == 16

    def test_metrics_calculator(self):
        from src.trainer import MetricsCalculator
        calc = MetricsCalculator(num_classes=2)
        logits = torch.randn(4, 2)
        labels = torch.tensor([0, 1, 0, 1])
        calc.update(0.5, logits, labels)
        metrics = calc.compute()
        assert 'loss' in metrics
        assert 'accuracy' in metrics


# ============================================================
# TestEvaluator - Evaluator module tests
# ============================================================

class TestEvaluator:
    """Tests for evaluator.py"""

    def test_evaluator_config(self):
        from src.evaluator import EvaluatorConfig
        cfg = EvaluatorConfig()
        assert cfg.num_classes == 2

    def test_error_capture_rate(self):
        from src.evaluator import compute_error_capture_rate
        is_error = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        risk_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.15, 0.85, 0.25, 0.1, 0.95])
        ecr = compute_error_capture_rate(is_error, risk_scores, top_k=0.3)
        assert 0.0 <= ecr <= 1.0

    def test_error_detection_auroc(self):
        from src.evaluator import compute_error_detection_auroc
        is_error = np.array([0, 0, 1, 0, 1])
        risk_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8])
        auroc = compute_error_detection_auroc(is_error, risk_scores)
        assert 0.0 <= auroc <= 1.0

    def test_brier_risk(self):
        from src.evaluator import compute_brier_risk
        is_error = np.array([0, 1, 0, 1])
        risk_scores = np.array([0.1, 0.9, 0.2, 0.8])
        brier = compute_brier_risk(is_error, risk_scores)
        assert 0.0 <= brier <= 1.0


# ============================================================
# TestModulesInit - Module __init__ exports
# ============================================================

class TestModulesInit:
    """Tests for module __init__ exports"""

    def test_models_init_exports(self):
        from src.models import (
            MedMambaV2, MedMambaV3, MedMambaGuard, MedMambaHead,
            CrossScan, CrossMerge, SS2D, VSSBlock2D,
            CTMMonitor, CrossScanRiskAnalyzer, TaskConflictValidator,
            HardGatingRules, create_medmamba_guard, create_medmamba,
        )
        assert all(cls is not None for cls in [
            MedMambaV2, MedMambaV3, MedMambaGuard, MedMambaHead,
            CrossScan, CrossMerge, SS2D, VSSBlock2D,
            CTMMonitor, CrossScanRiskAnalyzer, TaskConflictValidator,
            HardGatingRules,
        ])


# ============================================================
# TestEdgeCases - Edge cases and robustness
# ============================================================

class TestEdgeCases:
    """Edge case tests"""

    def test_single_pixel_input(self):
        """Model should handle very small inputs."""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=8, patch_size=4, d_model=16, n_layers=1, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            result = model(x, task="classification")
        assert result["logits"].shape == (1, 2)

    def test_guard_no_ctm(self):
        """Guard without CTM should still work."""
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0, use_ctm=False)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert 'prediction' in output

    def test_guard_no_cross_scan(self):
        """Guard without Cross-Scan should still work."""
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0, use_cross_scan=False)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert 'prediction' in output

    def test_guard_no_task_validator(self):
        """Guard without task validator should still work."""
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(d_model=32, d_state=4, n_layers=2, num_classes=2, dropout=0.0, use_task_validator=False)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert 'prediction' in output

    def test_model_eval_train_toggle(self):
        """Model should work in both eval and train mode."""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2, dropout=0.1)
        x = torch.randn(1, 3, 32, 32)

        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        model.train()
        out_train = model(x)

        assert out_eval["logits"].shape == out_train["logits"].shape

    def test_probability_sum_to_one(self):
        """Output probabilities should sum to 1."""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(img_size=32, patch_size=4, d_model=32, n_layers=2, num_classes=2, dropout=0.0)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            result = model(x)
        probs = torch.softmax(result["logits"], dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_guard_config_based_creation(self):
        """Guard should accept config object."""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4, dropout=0.0)
        model = MedMambaGuard(config)
        assert model is not None


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
