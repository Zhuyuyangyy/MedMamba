"""
tests/test_guard_core.py
MedMamba-Guard V0.1 核心风险机制单元测试

运行: pytest tests/test_guard_core.py -v
依赖: pip install pytest torch medmnist

验收项:
  TestCTMMonitor:
    - test_smooth_vs_jump_sequence: 平滑序列risk < 跳变序列risk
    - test_ctm_metrics_nonempty: CTM指标非零非NaN
    - test_overconfidence_detection: 高conf+高state_risk应触发警告

  TestCrossScanRiskAnalyzer:
    - test_consistent_features_low_risk: 四方向一致 → risk低
    - test_divergent_features_high_risk: 一方向偏离 → risk升高
    - test_risk_l2_vs_cos_combination: 融合风险两种计算方式一致性

  TestTaskConflictValidator:
    - test_high_confidence_large_lesion: 高P+大病灶 → PASS
    - test_low_confidence_no_lesion: 低P+无病灶 → PASS
    - test_high_confidence_no_lesion: 高P+无病灶 → REVIEW
    - test_low_confidence_large_lesion: 低P+大病灶 → REVIEW
    - test_conflict_score_calculation: 冲突分数计算正确性

  TestMedMambaGuardIntegration:
    - test_forward_output_schema: 前向输出包含所有必需字段
    - test_risk_score_range: risk_score在合理范围[0, ~1.5]
    - test_risk_heatmap_shape: 热力图尺寸与输入对应
    - test_review_regions_structure: 复核区域格式正确
    - test_audit_log_contains_ctm: 审计日志包含CTM指标
"""

import sys
import pytest
import torch

sys.path.insert(0, '.')


# ──────────────────────────────────────────────────────────────
# TestCTMMonitor
# ──────────────────────────────────────────────────────────────

class TestCTMMonitor:
    """CTM状态轨迹监控器单元测试"""

    def setup_method(self):
        from src.models.ctm_monitor import CTMMonitor
        self.Monitor = CTMMonitor
        self.hidden_dim = 64
        self.monitor = self.Monitor(hidden_dim=self.hidden_dim)

    def test_smooth_vs_jump_sequence(self):
        """平滑序列risk < 跳变序列risk"""
        seq_len = 8

        # 平滑: 小随机游走
        smooth = []
        h = torch.randn(1, self.hidden_dim) * 0.5
        for _ in range(seq_len):
            smooth.append(h + torch.randn_like(h) * 0.05)
            h = smooth[-1]

        # 跳变: 第4步后剧烈变化
        jump = []
        h = torch.randn(1, self.hidden_dim) * 0.5
        for i in range(seq_len):
            if i >= 4:
                jump.append(h + torch.randn_like(h) * 3.0)  # 大幅跳变
            else:
                jump.append(h + torch.randn_like(h) * 0.05)
            h = jump[-1]

        conf = torch.tensor([0.8])

        m_smooth = self.monitor.compute_ctm_metrics(smooth, smooth, conf)
        m_jump = self.monitor.compute_ctm_metrics(jump, jump, conf)

        # 提取state_transition_instability
        v_smooth = m_smooth.get('state_transition_instability', torch.tensor(0.0)).mean().item()
        v_jump = m_jump.get('state_transition_instability', torch.tensor(0.0)).mean().item()

        assert v_jump > v_smooth, (
            f"跳变序列risk({v_jump:.4f})应该大于平滑序列risk({v_smooth:.4f})"
        )

    def test_ctm_metrics_nonempty(self):
        """CTM指标计算结果非零非NaN"""
        seq_len = 6
        hidden_states = [torch.randn(2, self.hidden_dim) for _ in range(seq_len)]
        delta_states = [torch.randn(2, self.hidden_dim) for _ in range(seq_len)]
        conf = torch.tensor([0.7, 0.85])

        metrics = self.monitor.compute_ctm_metrics(hidden_states, delta_states, conf)

        required_keys = [
            'state_transition_instability',
            'input_response_drift',
            'trajectory_convergence',
            'overconfidence_risk',
        ]

        for k in required_keys:
            assert k in metrics, f"missing CTM metric: {k}"
            v = metrics[k]
            assert isinstance(v, torch.Tensor), f"{k} should be tensor, got {type(v)}"
            assert not torch.isnan(v).any(), f"{k} contains NaN"
            assert not torch.isinf(v).any(), f"{k} contains inf"

        risk = self.monitor.get_state_risk_score()
        assert 0.0 <= risk <= 10.0, f"state_risk_score out of range: {risk}"

    def test_overconfidence_detection(self):
        """高conf + 高state_risk → 高overconfidence_risk"""
        # 高置信但隐藏态跳变剧烈
        seq_len = 6
        jump = []
        h = torch.randn(1, self.hidden_dim)
        for i in range(seq_len):
            if i >= 3:
                jump.append(h + torch.randn_like(h) * 2.5)
            else:
                jump.append(h + torch.randn_like(h) * 0.05)
            h = jump[-1]

        high_conf = torch.tensor([0.97])  # 极高置信
        metrics = self.monitor.compute_ctm_metrics(jump, jump, high_conf)

        oc_risk = metrics.get('overconfidence_risk', torch.tensor(0.0)).item()
        assert oc_risk > 0.3, (
            f"高conf(0.97)+高状态波动应该产生较高overconfidence_risk，"
            f"实际={oc_risk:.4f}"
        )


# ──────────────────────────────────────────────────────────────
# TestCrossScanRiskAnalyzer
# ──────────────────────────────────────────────────────────────

class TestCrossScanRiskAnalyzer:
    """Cross-Scan方向一致性风险分析器单元测试"""

    def setup_method(self):
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        self.Analyzer = CrossScanRiskAnalyzer
        self.feature_dim = 64
        self.analyzer = self.Analyzer(feature_dim=self.feature_dim)
        self.B, self.H, self.W = 2, 8, 8
        self.n_patches = self.H * self.W

    def _make_features(self, scale=1.0, divergence_idx=None):
        """生成四个方向特征，divergence_idx指定哪个方向偏离"""
        base = torch.randn(self.B, self.feature_dim, self.n_patches) * scale
        features = []
        for k in range(4):
            if divergence_idx is not None and k == divergence_idx:
                feat = torch.randn(self.B, self.feature_dim, self.n_patches) * (scale * 4.0)
            else:
                feat = base.clone()
            features.append(feat)
        return features

    def test_consistent_features_low_risk(self):
        """四方向完全一致 → risk低"""
        features = self._make_features(scale=1.0, divergence_idx=None)
        result = self.analyzer.compute_divergence_maps(features)

        risk_l2 = result.get('risk_l2', torch.tensor(0.0)).mean().item()
        risk_cos = result.get('risk_cos', torch.tensor(0.0)).mean().item()

        assert risk_l2 <= 0.5, f"一致特征risk_l2应较小，实际={risk_l2:.4f}"
        assert risk_cos <= 0.5, f"一致特征risk_cos应较小，实际={risk_cos:.4f}"

    def test_divergent_features_high_risk(self):
        """一方向明显偏离 → risk升高"""
        features_consistent = self._make_features(scale=1.0, divergence_idx=None)
        features_divergent = self._make_features(scale=1.0, divergence_idx=2)

        r_c = self.analyzer.compute_divergence_maps(features_consistent)['risk_l2'].mean().item()
        r_d = self.analyzer.compute_divergence_maps(features_divergent)['risk_l2'].mean().item()

        assert r_d > r_c, (
            f"分歧特征risk({r_d:.4f})应该明显高于一致特征risk({r_c:.4f})"
        )

    def test_risk_l2_vs_cos_combination(self):
        """融合risk计算: L2和Cos在分歧情况下同向变化"""
        features = self._make_features(scale=1.0, divergence_idx=1)
        result = self.analyzer.compute_divergence_maps(features)

        assert 'risk_l2' in result and 'risk_cos' in result
        assert 'risk_fused' in result

        r_l2 = result['risk_l2'].mean().item()
        r_cos = result['risk_cos'].mean().item()
        r_fused = result['risk_fused'].mean().item()

        # 融合risk应介于L2和Cos之间或大于两者
        assert r_fused >= min(r_l2, r_cos) * 0.5, f"fused risk({r_fused}) unreasonable"
        assert r_fused <= max(r_l2, r_cos) * 2.0, f"fused risk({r_fused}) unreasonable"

    def test_get_scan_risk_score(self):
        """get_scan_risk_score返回合理范围的标量"""
        features = self._make_features(scale=2.0, divergence_idx=0)
        score = self.analyzer.get_scan_risk_score(features)
        assert isinstance(score, float), f"score should be float, got {type(score)}"
        assert 0.0 <= score <= 5.0, f"scan_risk_score out of range: {score}"

    def test_get_inconsistent_regions(self):
        """get_inconsistent_regions返回有效bbox列表"""
        features = self._make_features(scale=1.0, divergence_idx=2)
        regions = self.analyzer.get_inconsistent_regions(threshold=0.2)

        assert isinstance(regions, list)
        for r in regions:
            assert 'bbox' in r, "region should have bbox"
            assert 'risk_type' in r, "region should have risk_type"
            assert 'risk_level' in r, "region should have risk_level"
            bbox = r['bbox']
            assert len(bbox) == 4, f"bbox should be [x1,y1,x2,y2], got {bbox}"


# ──────────────────────────────────────────────────────────────
# TestTaskConflictValidator
# ──────────────────────────────────────────────────────────────

class TestTaskConflictValidator:
    """分类-分割互证门控单元测试"""

    def setup_method(self):
        from src.models.task_conflict_validator import TaskConflictValidator
        self.validator = TaskConflictValidator()

    def _cls(self, p):
        return torch.tensor([[p]])

    def _seg(self, value, size=8):
        return torch.ones(1, 1, size, size) * value

    def test_high_confidence_large_lesion(self):
        """高分类概率+大病灶区域 → PASS"""
        result = self.validator.validate(self._cls(0.9), self._seg(0.85))
        assert result['action'] == 'PASS', f"expected PASS, got {result['action']}"

    def test_low_confidence_no_lesion(self):
        """低分类概率+无病灶区域 → PASS"""
        result = self.validator.validate(self._cls(0.08), self._seg(0.0))
        assert result['action'] == 'PASS', f"expected PASS, got {result['action']}"

    def test_high_confidence_no_lesion(self):
        """高分类概率+无明显病灶 → REVIEW (分类过度自信)"""
        result = self.validator.validate(self._cls(0.95), self._seg(0.01))
        assert result['action'] in ('REVIEW', 'CONFLICT_OVER_CONFIDENT'), (
            f"expected REVIEW/CONFLICT_OVER_CONFIDENT, got {result['action']}"
        )

    def test_low_confidence_large_lesion(self):
        """低分类概率+明显病灶 → REVIEW (漏诊风险)"""
        result = self.validator.validate(self._cls(0.07), self._seg(0.88))
        assert result['action'] in ('REVIEW', 'CONFLICT_LESION_MISSED'), (
            f"expected REVIEW/CONFLICT_LESION_MISSED, got {result['action']}"
        )

    def test_conflict_score_calculation(self):
        """冲突分数S_conflict计算正确性"""
        # 高P但分割几乎为空 → 高冲突
        result_high = self.validator.validate(self._cls(0.9), self._seg(0.0))
        conflict_high = result_high.get('conflict_score', 0.0)

        # 高P且分割大块 → 低冲突
        result_low = self.validator.validate(self._cls(0.9), self._seg(0.85))
        conflict_low = result_low.get('conflict_score', 0.0)

        assert conflict_high > conflict_low, (
            f"高P+无病灶冲突分({conflict_high:.4f})应大于高P+大病灶({conflict_low:.4f})"
        )

    def test_seg_evidence_breakdown(self):
        """分割空间证据E_seg的三个分量都存在"""
        result = self.validator.validate(self._cls(0.75), self._seg(0.6))

        assert 'seg_evidence' in result
        ev = result['seg_evidence']
        assert 'area_score' in ev, "E_seg应包含area_score"
        assert 'compactness_score' in ev, "E_seg应包含compactness_score"
        assert 'boundary_stability' in ev, "E_seg应包含boundary_stability"

        # 值应在合理范围
        for k, v in ev.items():
            assert 0.0 <= v <= 1.5, f"{k} value {v} out of range"


# ──────────────────────────────────────────────────────────────
# TestMedMambaGuardIntegration
# ──────────────────────────────────────────────────────────────

class TestMedMambaGuardIntegration:
    """MedMambaGuard主模型集成测试"""

    def setup_method(self):
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig
        self.Model = MedMambaGuard
        self.Config = MedMambaGuardConfig
        self.config = self.Config(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4,
            dropout=0.0,
        )
        self.model = self.Model(self.config)
        self.model.eval()

    def test_forward_output_schema(self):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)

        required_fields = [
            'prediction', 'confidence', 'risk_score', 'risk_components',
            'risk_heatmap', 'review_regions', 'action', 'audit_log'
        ]

        for field in required_fields:
            assert field in output, f"missing output field: {field}"

    def test_risk_score_range(self):
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        rs = output['risk_score']
        assert 0.0 <= rs <= 2.0, f"risk_score {rs} out of [0, 2.0]"

    def test_risk_heatmap_shape(self):
        B, C, H, W = 2, 3, 32, 32
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            output = self.model(x)
        rhm = output['risk_heatmap']

        assert isinstance(rhm, torch.Tensor), f"risk_heatmap should be tensor, got {type(rhm)}"
        assert rhm.shape[-2:] == (H, W), (
            f"risk_heatmap spatial shape {rhm.shape[-2:]} should match input {(H,W)}"
        )

    def test_review_regions_structure(self):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        regions = output['review_regions']

        assert isinstance(regions, list), f"review_regions should be list, got {type(regions)}"

        for r in regions:
            assert 'bbox' in r, f"region missing bbox: {r}"
            assert 'risk_type' in r, f"region missing risk_type: {r}"
            assert 'risk_level' in r, f"region missing risk_level: {r}"
            bbox = r['bbox']
            assert len(bbox) == 4, f"bbox should be [x1,y1,x2,y2], got {len(bbox)} elements"

    def test_audit_log_contains_ctm(self):
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = self.model(x)
        audit = output['audit_log']

        assert isinstance(audit, dict), f"audit_log should be dict, got {type(audit)}"
        assert 'ctm_metrics' in audit, "audit_log should contain ctm_metrics"
        assert 'scan_metrics' in audit, "audit_log should contain scan_metrics"
        assert 'model_version' in audit, "audit_log should contain model_version"
        assert 'timestamp' in audit, "audit_log should contain timestamp"

    def test_hard_gating_rules_doctor_review(self):
        action = self.model.hard_gating_rules(
            confidence=0.6,
            r_state=0.3,
            r_scan=0.3,
            r_task=0.3,
            r_entropy=0.3,
            r_total=0.85
        )
        assert action in ('doctor_review', 'overconfidence_warning'), (
            f"high R_total should trigger review, got {action}"
        )

    def test_hard_gating_rules_low_risk_passes(self):
        action = self.model.hard_gating_rules(
            confidence=0.75,
            r_state=0.1,
            r_scan=0.1,
            r_task=0.05,
            r_entropy=0.1,
            r_total=0.2
        )
        assert action == 'PASS', f"low risk should PASS, got {action}"

    def test_batch_processing(self):
        B = 2
        x = torch.randn(B, 3, 32, 32)
        with torch.no_grad():
            outputs = self.model(x)

        assert outputs['risk_heatmap'].shape[0] == B, (
            f"batch risk_heatmap B dim {outputs['risk_heatmap'].shape[0]} should be {B}"
        )


# ──────────────────────────────────────────────────────────────
# 运行入口
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])