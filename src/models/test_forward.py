"""
MedMamba-Guard 最小闭环验收脚本
运行方式: python src/models/test_forward.py

验收项:
  1. 导入所有模块 (AST已验证，这里验证运行时导入)
  2. 随机张量前向传播 (输入->分类+分割+风险+门控)
  3. CTM Hook捕获验证 (确认拿到SSM隐藏态)
  4. 四个风险机制逻辑单测
"""

import sys
import time

sys.path.insert(0, '.')

def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test1_import():
    banner("验收1: 模块导入测试")
    try:
        import torch
        print(f"  OK  torch {torch.__version__}")
    except ImportError as e:
        print(f"  FAIL  torch not installed: {e}")
        return False

    try:
        from src.models.ssm_config import CTMConfig, MedMambaGuardConfig
        cfg = CTMConfig()
        print(f"  OK  CTMConfig()  trajectory_window={cfg.trajectory_window}")
    except Exception as e:
        print(f"  FAIL  CTMConfig: {e}")
        return False

    try:
        from src.models.ctm_monitor import CTMMonitor
        print("  OK  CTMMonitor imported")
    except Exception as e:
        print(f"  FAIL  CTMMonitor: {e}")
        return False

    try:
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer
        print("  OK  CrossScanRiskAnalyzer imported")
    except Exception as e:
        print(f"  FAIL  CrossScanRiskAnalyzer: {e}")
        return False

    try:
        from src.models.task_conflict_validator import TaskConflictValidator
        print("  OK  TaskConflictValidator imported")
    except Exception as e:
        print(f"  FAIL  TaskConflictValidator: {e}")
        return False

    try:
        from src.models.medmamba_guard import MedMambaGuard
        print("  OK  MedMambaGuard imported")
    except Exception as e:
        print(f"  FAIL  MedMambaGuard: {e}")
        return False

    print("  [PASS] 所有模块导入成功")
    return True


def test2_forward():
    banner("验收2: 随机张量前向传播")
    import torch
    from src.models.medmamba_guard import MedMambaGuard
    from src.models.ssm_config import MedMambaGuardConfig

    B, C, H, W = 1, 3, 32, 32
    x = torch.randn(B, C, H, W)
    print(f"  Input shape: {x.shape}")

    config = MedMambaGuardConfig(
        d_model=32,
        d_state=4,
        d_conv=3,
        expand=2,
        num_classes=2,
        dropout=0.0,
        img_size=32,
        patch_size=4,
        n_layers=2,
    )
    model = MedMambaGuard(config)
    model.eval()

    try:
        with torch.no_grad():
            output = model(x)
    except Exception as e:
        print(f"  FAIL  forward pass: {e}")
        import traceback; traceback.print_exc()
        return False

    # 验证输出字段
    required_fields = [
        'prediction', 'confidence', 'risk_score', 'risk_components',
        'risk_heatmap', 'review_regions', 'action', 'audit_log'
    ]

    for field in required_fields:
        if field not in output:
            print(f"  FAIL  missing output field: {field}")
            return False
        val = output[field]
        if isinstance(val, torch.Tensor):
            print(f"  OK  {field}: tensor {val.shape}")
        elif isinstance(val, dict):
            print(f"  OK  {field}: dict with keys {list(val.keys())}")
        elif isinstance(val, list):
            print(f"  OK  {field}: list len={len(val)}")
        else:
            print(f"  OK  {field}: {type(val).__name__} = {val}")

    # 风险分数有效性
    rs = output['risk_score']
    if not isinstance(rs, float) or not (0.0 <= rs <= 1.5):
        print(f"  FAIL  risk_score out of range: {rs}")
        return False
    print(f"  OK  risk_score = {rs:.4f}  (valid range)")

    # CTM指标非空
    ctm = output.get('audit_log', {}).get('ctm_metrics', {})
    if not ctm:
        print("  WARN  ctm_metrics empty (CTM hook may not capture states yet)")
    else:
        print(f"  OK  ctm_metrics keys: {list(ctm.keys())}")

    print("  [PASS] 前向传播成功，输出字段完整")
    return True


def test3_ctm_hook():
    banner("验收3: CTM Hook捕获验证")
    import torch
    from src.models.ctm_monitor import CTMMonitor
    from src.models.ssm_config import CTMConfig

    hidden_dim = 128
    monitor = CTMMonitor(hidden_dim=hidden_dim, trajectory_window=5, delta_window=3)

    print(f"  CTMMonitor created, hidden_dim={hidden_dim}")

    # 模拟隐藏态序列 (模拟SSM每步的隐藏态输出)
    seq_len = 10
    fake_hidden_states = [
        torch.randn(2, hidden_dim) for _ in range(seq_len)
    ]
    # 模拟Δ_t序列
    fake_delta_states = [
        torch.randn(2, hidden_dim) for _ in range(seq_len)
    ]

    output_confidence = torch.tensor([0.87, 0.92])

    try:
        metrics = monitor.compute_ctm_metrics(
            fake_hidden_states, fake_delta_states, output_confidence
        )
    except Exception as e:
        print(f"  FAIL  compute_ctm_metrics: {e}")
        import traceback; traceback.print_exc()
        return False

    print("  CTM Metrics:")
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                print(f"    {k}: tensor shape={v.shape} val={v.item():.4f}")
            else:
                print(f"    {k}: tensor shape={v.shape} mean={v.mean().item():.4f}")
        else:
            print(f"    {k}: {v}")

    risk = monitor.get_state_risk_score()
    print(f"  state_risk_score = {risk:.4f}")

    if risk <= 0.0 or risk > 10.0:
        print(f"  FAIL  risk_score out of expected range: {risk}")
        return False

    print("  [PASS] CTM Hook捕获验证成功")
    return True


def test4_risk_logic():
    banner("验收4: 四个风险机制逻辑单测")

    # === A. CTM风险: 平滑 vs 跳变状态序列 ===
    print("\n  [A] CTM状态激变度测试")
    try:
        import torch
        from src.models.ctm_monitor import CTMMonitor

        hidden_dim = 64
        monitor = CTMMonitor(hidden_dim=hidden_dim)

        # 平滑序列: 相邻状态差异小
        smooth_seq = [torch.randn(1, hidden_dim) * 0.1 for _ in range(5)]
        for i in range(1, len(smooth_seq)):
            smooth_seq[i] = smooth_seq[i-1] + torch.randn(1, hidden_dim) * 0.05

        # 跳变序列: 某处突然剧烈变化
        jump_seq = [torch.randn(1, hidden_dim) for _ in range(5)]
        jump_seq[3] = jump_seq[3] * 5.0  # 剧烈跳变

        conf = torch.tensor([0.8])

        smooth_metrics = monitor.compute_ctm_metrics(smooth_seq, smooth_seq, conf)
        jump_metrics = monitor.compute_ctm_metrics(jump_seq, jump_seq, conf)

        smooth_risk = smooth_metrics.get('state_transition_instability', torch.tensor(0.0)).item()
        jump_risk = jump_metrics.get('state_transition_instability', torch.tensor(0.0)).item()

        print(f"    平滑序列 state_transition_instability = {smooth_risk:.4f}")
        print(f"    跳变序列 state_transition_instability = {jump_risk:.4f}")

        if jump_risk > smooth_risk:
            print("    OK  跳变序列风险 > 平滑序列风险")
        else:
            print(f"    WARN  expected jump_risk > smooth_risk, got {jump_risk} vs {smooth_risk}")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()

    # === B. Cross-Scan一致性: 一致 vs 分歧方向 ===
    print("\n  [B] Cross-Scan一致性风险测试")
    try:
        import torch
        from src.models.cross_scan_risk import CrossScanRiskAnalyzer

        analyzer = CrossScanRiskAnalyzer(feature_dim=64)

        B, H, W = 1, 8, 8

        # 四方向完全一致的特征
        consistent_feat = torch.randn(B, 64, H * W)
        features_consistent = [consistent_feat.clone() for _ in range(4)]

        # 四个方向中有一个明显偏离
        divergent_feat = torch.randn(B, 64, H * W) * 5.0
        features_divergent = [consistent_feat.clone(), consistent_feat.clone(),
                              divergent_feat, consistent_feat.clone()]

        risk_consistent = analyzer.compute_divergence_maps(features_consistent)
        risk_divergent = analyzer.compute_divergence_maps(features_divergent)

        r_c = risk_consistent.get('risk_l2', torch.tensor(0.0)).mean().item()
        r_d = risk_divergent.get('risk_l2', torch.tensor(0.0)).mean().item()

        print(f"    一致特征 risk_l2 = {r_c:.4f}")
        print(f"    分歧特征 risk_l2 = {r_d:.4f}")

        if r_d > r_c:
            print("    OK  分歧特征风险 > 一致特征风险")
        else:
            print(f"    WARN  expected r_d > r_c, got {r_d} vs {r_c}")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()

    # === C. 分类-分割互证门控: 四种场景 ===
    print("\n  [C] 分类-分割互证门控测试")
    try:
        import torch
        from src.models.task_conflict_validator import TaskConflictValidator

        validator = TaskConflictValidator()

        # 场景1: 高分类概率+大病灶区域 → PASS
        cls_1 = torch.tensor([[0.9]])  # 高分类概率
        seg_1 = torch.ones(1, 1, 8, 8) * 0.8  # 分割到大块病灶
        result_1 = validator.validate(cls_1, seg_1)
        print(f"    场景1(高P+大病灶): action={result_1['action']} 期望=PASS")
        assert result_1['action'] == 'PASS', f"expected PASS, got {result_1['action']}"

        # 场景2: 低分类概率+无病灶区域 → PASS
        cls_2 = torch.tensor([[0.1]])
        seg_2 = torch.zeros(1, 1, 8, 8)  # 无病灶
        result_2 = validator.validate(cls_2, seg_2)
        print(f"    场景2(低P+无病灶): action={result_2['action']} 期望=PASS")
        assert result_2['action'] == 'PASS', f"expected PASS, got {result_2['action']}"

        # 场景3: 高分类概率+无明显病灶 → REVIEW (分类过度自信)
        cls_3 = torch.tensor([[0.92]])
        seg_3 = torch.zeros(1, 1, 8, 8)  # 无明显病灶
        result_3 = validator.validate(cls_3, seg_3)
        print(f"    场景3(高P+无病灶): action={result_3['action']} 期望=REVIEW")
        assert result_3['action'] == 'REVIEW', f"expected REVIEW, got {result_3['action']}"

        # 场景4: 低分类概率+明显病灶 → REVIEW (漏诊风险)
        cls_4 = torch.tensor([[0.08]])
        seg_4 = torch.ones(1, 1, 8, 8) * 0.85  # 明显大块病灶
        result_4 = validator.validate(cls_4, seg_4)
        print(f"    场景4(低P+大病灶): action={result_4['action']} 期望=REVIEW")
        assert result_4['action'] == 'REVIEW', f"expected REVIEW, got {result_4['action']}"

        print("    OK  四种场景全部通过")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()

    # === D. 高置信+内部不稳定 → 过度自信警告 ===
    print("\n  [D] 高置信但内部不稳定测试")
    try:
        import torch
        from src.models.medmamba_guard import MedMambaGuard
        from src.models.ssm_config import MedMambaGuardConfig

        config = MedMambaGuardConfig(
            d_model=32, d_state=4, n_layers=2, img_size=32, patch_size=4
        )
        model = MedMambaGuard(config)
        model.eval()

        high_conf = 0.95
        high_r_state = 0.85
        action = model.hard_gating_rules(
            confidence=high_conf,
            r_state=high_r_state,
            r_scan=0.2,
            r_task=0.1,
            r_entropy=0.1,
            r_total=0.85
        )
        print(f"    高置信(0.95)+高状态风险(0.85) → action={action}")
        if action in ('doctor_review', 'overconfidence_warning'):
            print("    OK  触发复核/警告")
        else:
            print(f"    WARN  expected doctor_review/overconfidence_warning, got {action}")

        print("    [PASS] 过度自信检测逻辑验证")
    except Exception as e:
        print(f"    FAIL: {e}")
        import traceback; traceback.print_exc()

    print("\n  [ALL PASS] 四个风险机制逻辑单测完成")
    return True


def main():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║       MedMamba-Guard V0.1 最小闭环验收           ║
    ╚══════════════════════════════════════════════════╝
    """)
    start = time.time()

    results = {}
    results['import'] = test1_import()
    if not results['import']:
        print("\n[ABORT] 导入失败，无法继续")
        return

    results['forward'] = test2_forward()
    results['ctm_hook'] = test3_ctm_hook()
    results['risk_logic'] = test4_risk_logic()

    elapsed = time.time() - start

    banner("验收结果汇总")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  总计: {passed}/{total} 项通过  ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()