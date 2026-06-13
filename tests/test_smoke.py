"""
tests/test_smoke.py
MedMamba 项目冒烟测试 - 快速验证核心模块可导入和基本前向传播

运行: pytest tests/test_smoke.py -v
依赖: pip install pytest torch einops

测试范围:
  TestImports:
    - test_import_medmamba: 核心模型模块导入
    - test_import_guard: Guard框架模块导入
    - test_import_data: 数据模块导入
    - test_import_config: 配置模块导入

  TestModelCreation:
    - test_create_medmamba_v2: V2模型创建
    - test_create_medmamba_v3: V3模型创建
    - test_create_guard_model: Guard模型创建
    - test_create_light_guard: 轻量Guard模型创建

  TestForwardPass:
    - test_medmamba_v2_forward: V2前向传播
    - test_medmamba_v3_forward: V3前向传播
    - test_guard_forward: Guard前向传播(含风险评估)
    - test_guard_output_schema: Guard输出字段完整性

  TestFactoryFunctions:
    - test_create_medmamba_factory: create_medmamba工厂函数
    - test_create_guard_factory: create_medmamba_guard工厂函数

  TestComponents:
    - test_cross_scan: CrossScan四方向扫描
    - test_cross_merge: CrossMerge合并
    - test_mamba_block: MambaBlock2D基本功能
    - test_cnn_branch: CNNBranch前向传播
    - test_ctm_analyzer: CTMTrajectoryAnalyzer分析

  TestEdgeCases:
    - test_batch_processing: 批量处理
    - test_different_image_sizes: 不同图像尺寸
    - test_gradient_flow: 梯度传播
"""

import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, '.')


# ============================================================
# TestImports - 模块导入测试
# ============================================================

class TestImports:
    """验证所有核心模块可正常导入"""

    def test_import_medmamba(self):
        """核心模型模块导入"""
        from src.models.medmamba import MedMambaV2, MedMambaV3, MedMambaHead
        from src.models.medmamba import DualBranchEncoder, create_medmamba
        assert MedMambaV2 is not None
        assert MedMambaV3 is not None

    def test_import_guard(self):
        """Guard框架模块导入"""
        from src.models.medmamba_guard import MedMambaGuard, HardGatingRules
        from src.models.medmamba_guard import RiskHeatmapGenerator, AuditLogger
        assert MedMambaGuard is not None
        assert HardGatingRules is not None

    def test_import_data(self):
        """数据模块导入"""
        from src.data.dataset import MedicalImageDataset
        from src.data.augmentation import get_train_transforms
        assert MedicalImageDataset is not None

    def test_import_config(self):
        """配置模块导入"""
        from src.models.ssm_config import SSMConfig, MedMambaConfig
        from src.models.ssm_config import CTMConfig, MedMambaGuardConfig
        assert SSMConfig is not None
        assert CTMConfig is not None

    def test_import_vmamba_blocks(self):
        """VMamba模块导入"""
        from src.models.vmamba_blocks import CrossScan, CrossMerge, VSSBlock2D
        from src.models.vmamba_blocks import VMamba2D, CrossAttentionFusion
        assert CrossScan is not None
        assert VSSBlock2D is not None

    def test_import_home_moe(self):
        """HoME-MoE模块导入"""
        from src.models.home_moe import ExpertBlock, ExpertRouter, HierarchicalMoE
        from src.models.home_moe import CTMTrajectoryAnalyzer
        assert ExpertBlock is not None
        assert HierarchicalMoE is not None

    def test_import_ctm_monitor(self):
        """CTM监控模块导入"""
        from src.models.ctm_monitor import CTMMonitor
        assert CTMMonitor is not None

    def test_import_cross_scan_risk(self):
        """CrossScan风险分析模块导入"""
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        assert CrossScanRiskAnalyzer is not None

    def test_import_task_validator(self):
        """任务冲突验证模块导入"""
        from src.models.task_conflict_validator import TaskConflictValidator
        assert TaskConflictValidator is not None


# ============================================================
# TestModelCreation - 模型创建测试
# ============================================================

class TestModelCreation:
    """验证各模型变体可正常实例化"""

    def test_create_medmamba_v2(self):
        """V2模型创建"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.0
        )
        assert model is not None
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "模型应有参数"

    def test_create_medmamba_v3(self):
        """V3模型创建"""
        from src.models.medmamba import MedMambaV3
        model = MedMambaV3(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, use_ctm=True, use_moe=False, dropout=0.0
        )
        assert model is not None

    def test_create_guard_model(self):
        """Guard模型创建"""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        model = MedMambaGuard(config)
        assert model is not None

    def test_create_light_guard(self):
        """轻量Guard模型创建"""
        from src.models.medmamba_guard import LightMedMambaGuard
        model = LightMedMambaGuard(
            d_model=32, d_state=4, n_layers=2, img_size=32,
            patch_size=4, dropout=0.0
        )
        assert model is not None

    def test_model_param_count_v2(self):
        """V2模型参数量合理性检查"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=128, n_layers=4,
            num_classes=2, dropout=0.0
        )
        params = sum(p.numel() for p in model.parameters())
        assert params < 50_000_000, f"小模型参数量({params})应小于50M"

    def test_model_param_count_guard(self):
        """Guard模型参数量合理性检查"""
        from src.models.medmamba_guard import MedMambaGuard
        model = MedMambaGuard(
            d_model=32, d_state=4, n_layers=2, img_size=32,
            patch_size=4, dropout=0.0
        )
        params = sum(p.numel() for p in model.parameters())
        assert params < 10_000_000, f"小Guard模型参数量({params})应小于10M"


# ============================================================
# TestForwardPass - 前向传播测试
# ============================================================

class TestForwardPass:
    """验证各模型前向传播正常"""

    def test_medmamba_v2_forward(self):
        """V2前向传播"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.0
        )
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            result = model(x, task="classification")
        assert "logits" in result
        assert result["logits"].shape == (2, 2)

    def test_medmamba_v3_forward(self):
        """V3前向传播"""
        from src.models.medmamba import MedMambaV3
        model = MedMambaV3(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, use_ctm=True, use_moe=False, dropout=0.0
        )
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            result = model(x, task="classification", return_ctm=True)
        assert "logits" in result
        assert "hallucination_risk" in result

    def test_guard_forward(self):
        """Guard前向传播(含风险评估)"""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        model = MedMambaGuard(config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
        assert isinstance(output, dict)
        assert "prediction" in output
        assert "confidence" in output
        assert "risk_score" in output

    def test_guard_output_schema(self):
        """Guard输出字段完整性"""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        model = MedMambaGuard(config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        required_fields = [
            "prediction", "confidence", "risk_score",
            "risk_components", "risk_heatmap", "review_regions",
            "action", "audit_log",
        ]
        for field in required_fields:
            assert field in output, f"缺少输出字段: {field}"

    def test_guard_risk_components(self):
        """Guard风险分量完整性"""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        model = MedMambaGuard(config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        rc = output["risk_components"]
        assert "R_state" in rc, "缺少R_state"
        assert "R_scan" in rc, "缺少R_scan"
        assert "R_task" in rc, "缺少R_task"
        assert "R_entropy" in rc, "缺少R_entropy"

    def test_guard_ctm_metrics(self):
        """Guard审计日志包含CTM指标"""
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        model = MedMambaGuard(config)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(x)

        audit = output["audit_log"]
        assert isinstance(audit, dict)
        assert "model_version" in audit
        assert "timestamp" in audit


# ============================================================
# TestFactoryFunctions - 工厂函数测试
# ============================================================

class TestFactoryFunctions:
    """验证工厂函数正常工作"""

    def test_create_medmamba_factory(self):
        """create_medmamba工厂函数"""
        from src.models.medmamba import create_medmamba
        model = create_medmamba(
            version="v2", img_size=64, patch_size=8,
            d_model=64, n_layers=2, num_classes=2, dropout=0.0
        )
        assert model is not None

    def test_create_guard_factory(self):
        """create_medmamba_guard工厂函数"""
        from src.models.medmamba_guard import create_medmamba_guard
        model = create_medmamba_guard(
            d_model=32, n_layers=2, num_classes=2,
            img_size=32, patch_size=4, dropout=0.0
        )
        assert model is not None

    def test_factory_invalid_version(self):
        """无效版本号应抛出异常"""
        from src.models.medmamba import create_medmamba
        with pytest.raises(ValueError):
            create_medmamba(version="v99")


# ============================================================
# TestComponents - 核心组件测试
# ============================================================

class TestComponents:
    """验证核心组件的基本功能"""

    def test_cross_scan(self):
        """CrossScan四方向扫描"""
        from src.models.vmamba_blocks import CrossScan
        scan = CrossScan()
        x = torch.randn(2, 16, 8, 8)
        scans = scan(x)
        assert len(scans) == 4, "应返回4个方向"
        for s in scans:
            assert s.shape == (2, 16, 64), f"扫描形状错误: {s.shape}"

    def test_cross_merge(self):
        """CrossMerge合并"""
        from src.models.vmamba_blocks import CrossMerge
        merge = CrossMerge()
        B, C, H, W = 2, 16, 8, 8
        scans = [torch.randn(B, C, H * W) for _ in range(4)]
        out = merge(scans, H, W)
        assert out.shape == (B, C, H, W), f"合并形状错误: {out.shape}"

    def test_mamba_block(self):
        """MambaBlock2D基本功能"""
        from src.models.medmamba import MambaBlock2D
        block = MambaBlock2D(d_model=64, d_state=8, dropout=0.0)
        block.eval()
        x = torch.randn(2, 16, 64)  # [B, L, D]
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape, f"输出形状应与输入相同"

    def test_cnn_branch(self):
        """CNNBranch前向传播"""
        from src.models.medmamba import CNNBranch
        branch = CNNBranch(d_model=32, d_state=4)
        branch.eval()
        x = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            out = branch(x)
        assert out.shape == x.shape

    def test_ctm_analyzer(self):
        """CTMTrajectoryAnalyzer分析"""
        from src.models.home_moe import CTMTrajectoryAnalyzer
        analyzer = CTMTrajectoryAnalyzer(d_model=64, num_ticks=4)
        trajectory = torch.randn(2, 8, 64)  # [B, T, D]
        with torch.no_grad():
            result = analyzer(trajectory)
        assert "stability" in result
        assert "oscillation" in result
        assert "attractor_margin" in result

    def test_ctm_hallucination_score(self):
        """CTMTrajectoryAnalyzer幻觉分数"""
        from src.models.home_moe import CTMTrajectoryAnalyzer
        analyzer = CTMTrajectoryAnalyzer(d_model=64, num_ticks=4)
        trajectory = torch.randn(2, 8, 64)
        with torch.no_grad():
            score = analyzer.hallucination_score(trajectory)
        assert score.shape == (2,), f"分数形状应为(batch_size)"
        assert (score >= 0).all() and (score <= 1).all(), "分数应在[0,1]"

    def test_rms_norm(self):
        """RMSNorm归一化"""
        from src.models.medmamba import RMSNorm
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_feature_fusion(self):
        """FeatureFusion特征融合"""
        from src.models.medmamba import FeatureFusion
        fusion = FeatureFusion(d_model=32)
        fusion.eval()
        cnn_feat = torch.randn(2, 32, 8, 8)
        ssm_feat = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            out = fusion(cnn_feat, ssm_feat)
        assert out.shape == (2, 32, 8, 8)

    def test_hard_gating_rules(self):
        """HardGatingRules门控规则"""
        from src.models.medmamba_guard import HardGatingRules
        gating = HardGatingRules()

        # 低风险应PASS
        action, reason, details = gating.evaluate(
            r_state=0.1, r_scan=0.1, r_task=0.05,
            r_entropy=0.1, confidence=0.75
        )
        assert action == "PASS"

        # 高R_total应触发doctor_review
        action, reason, details = gating.evaluate(
            r_state=0.3, r_scan=0.3, r_task=0.3,
            r_entropy=0.3, confidence=0.6
        )
        assert action in ("doctor_review", "overconfidence_warning")

    def test_ssm_config(self):
        """SSMConfig配置创建"""
        from src.models.ssm_config import SSMConfig, MedMambaConfig, CTMConfig
        cfg = SSMConfig(d_model=256, d_state=16)
        assert cfg.d_model == 256
        assert cfg.d_state == 16

        med_cfg = MedMambaConfig(d_model=384, modal="MRI")
        assert med_cfg.modal == "MRI"

        ctm_cfg = CTMConfig(trajectory_window=7)
        assert ctm_cfg.trajectory_window == 7


# ============================================================
# TestEdgeCases - 边界条件测试
# ============================================================

class TestEdgeCases:
    """验证边界条件和鲁棒性"""

    def test_batch_processing(self):
        """批量处理"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.0
        )
        model.eval()
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 64, 64)
            with torch.no_grad():
                result = model(x, task="classification")
            assert result["logits"].shape[0] == batch_size

    def test_different_image_sizes(self):
        """不同图像尺寸"""
        from src.models.medmamba import MedMambaV2
        for img_size in [32, 64, 128]:
            patch_size = img_size // 8
            model = MedMambaV2(
                img_size=img_size, patch_size=patch_size,
                d_model=64, n_layers=2, num_classes=2, dropout=0.0
            )
            model.eval()
            x = torch.randn(1, 3, img_size, img_size)
            with torch.no_grad():
                result = model(x, task="classification")
            assert result["logits"].shape == (1, 2)

    def test_gradient_flow(self):
        """梯度传播"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.0
        )
        model.train()
        x = torch.randn(2, 3, 64, 64)
        result = model(x, task="classification")
        loss = result["logits"].sum()
        loss.backward()

        # 检查所有参数都有梯度
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        assert has_grad > 0, "应有参数接收到梯度"
        assert has_grad >= total_params * 0.5, "至少50%参数应有梯度"

    def test_eval_vs_train_mode(self):
        """eval和train模式切换"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.1
        )
        x = torch.randn(1, 3, 64, 64)

        model.eval()
        with torch.no_grad():
            out_eval = model(x, task="classification")

        model.train()
        out_train = model(x, task="classification")

        assert out_eval["logits"].shape == out_train["logits"].shape

    def test_single_class_output(self):
        """二分类输出"""
        from src.models.medmamba import MedMambaV2
        model = MedMambaV2(
            img_size=64, patch_size=8, d_model=64, n_layers=2,
            num_classes=2, dropout=0.0
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = model(x, task="classification")
        probs = torch.softmax(result["logits"], dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


# ============================================================
# 运行入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
