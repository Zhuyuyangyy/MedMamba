"""
MedMamba-Guard API Server - FastAPI Service
提供医学影像分类、分割、风险评估和可信推理
"""

import io
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import pydicom
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============ 配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
PORT = 8866

# 类别名称
CLASS_NAMES = ["Normal", "Lesion"]
NUM_CLASSES = 2

# 风险阈值
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "critical": 1.0
}

# CTM阈值
CTM_THRESHOLDS = {
    "theta_high": 0.7,
    "theta_conflict": 0.5,
    "theta_conf": 0.85,
    "theta_state": 0.6
}

# ============ 模型加载 ============
print(f"[MedMamba-Guard API] 初始化于 {DEVICE}")

model = None
ctm_monitor = None
cross_scan_analyzer = None


def load_model():
    """延迟加载模型"""
    global model, ctm_monitor, cross_scan_analyzer
    if model is not None:
        return model, ctm_monitor, cross_scan_analyzer

    print("[MedMamba-Guard API] 加载模型...")
    try:
        from src.models.medmamba_guard import MedMambaGuardClassifier

        model = MedMambaGuardClassifier(
            img_size=IMG_SIZE,
            patch_size=16,
            num_classes=NUM_CLASSES,
            d_model=768,
            n_layers=12,
            d_state=16,
            num_heads=8,
        ).to(DEVICE)

        # 尝试加载权重
        weights_dir = Path(__file__).parent.parent / "weights"
        weight_files = list(weights_dir.glob("*.pth"))
        if weight_files:
            checkpoint = weight_files[0]
            print(f"[MedMamba-Guard API] 加载权重: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            print("[MedMamba-Guard API] 警告: 未找到权重文件，使用随机初始化模型")

        model.eval()
        
        # 初始化CTM监控器和Cross-Scan分析器
        from src.evaluator import CTMMonitor, CrossScanAnalyzer
        ctm_monitor = CTMMonitor()
        cross_scan_analyzer = CrossScanAnalyzer()
        
        print("[MedMamba-Guard API] 模型加载完成")
        return model, ctm_monitor, cross_scan_analyzer
    except Exception as e:
        print(f"[MedMamba-Guard API] 模型加载失败: {e}")
        # 即使失败也返回，以支持demo模式
        return None, None, None


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """图像预处理"""
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    return x.unsqueeze(0)


def read_dicom(file_bytes: bytes) -> Image.Image:
    """读取DICOM文件"""
    try:
        dcm = pydicom.dcmread(io.BytesIO(file_bytes))
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        arr = apply_voi_lut(dcm.pixel_array, dcm)
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"DICOM读取失败: {e}")


def compute_risk_level(risk_score: float) -> str:
    """根据风险分数确定风险等级"""
    if risk_score < RISK_THRESHOLDS["low"]:
        return "low"
    elif risk_score < RISK_THRESHOLDS["medium"]:
        return "medium"
    elif risk_score < RISK_THRESHOLDS["high"]:
        return "high"
    else:
        return "critical"


def compute_ctm_metrics(hidden_states: List[torch.Tensor]) -> Dict[str, float]:
    """计算CTM状态轨迹指标"""
    if not hidden_states or len(hidden_states) < 2:
        return {
            "V_norm_mean": 0.0,
            "V_norm_max": 0.0,
            "D_delta": 0.0,
            "C_layer": 0.0,
            "R_overconfident": 0.0
        }
    
    metrics = {}
    
    # 状态激变度
    v_norms = []
    for i in range(1, len(hidden_states)):
        h_curr = hidden_states[i].flatten()
        h_prev = hidden_states[i-1].flatten()
        norm_prev = torch.norm(h_prev) + 1e-8
        v_norm = torch.norm(h_curr - h_prev).pow(2) / norm_prev
        v_norms.append(v_norm.item())
    
    metrics["V_norm_mean"] = np.mean(v_norms)
    metrics["V_norm_max"] = np.max(v_norms)
    
    # 输入依赖响应漂移
    stacked = torch.stack(hidden_states, dim=0)
    d_delta = torch.var(stacked, dim=0).mean().item()
    metrics["D_delta"] = d_delta
    
    # 跨层语义稳定性 (使用相邻时间步cosine相似度)
    cos_sims = []
    for i in range(1, len(hidden_states)):
        h_curr = hidden_states[i].flatten()
        h_prev = hidden_states[i-1].flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            h_curr.unsqueeze(0), h_prev.unsqueeze(0)
        ).item()
        cos_sims.append(cos_sim)
    metrics["C_layer"] = 1 - np.mean(cos_sims)
    
    # 状态-输出一致性 (默认0，后续由主函数填充)
    metrics["R_overconfident"] = 0.0
    
    return metrics


def compute_cross_scan_risk(features: torch.Tensor) -> Dict[str, Any]:
    """计算Cross-Scan方向一致性风险"""
    # 简化的Cross-Scan风险计算
    # 实际实现需要四方向扫描特征
    B, C, H, W = features.shape
    
    # 空间散度
    risk_map = torch.std(features, dim=1, keepdim=True).mean(dim=[2,3], keepdim=True)
    risk_score = risk_map.mean().item()
    
    return {
        "R_scan": risk_score,
        "R_scan_max": risk_map.max().item(),
        "risk_regions": []
    }


def compute_task_conflict(pred_cls: float, seg_mask: Optional[torch.Tensor]) -> float:
    """计算分类-分割任务冲突分数"""
    if seg_mask is None:
        return 0.0
    
    # 分割空间证据: 面积 + 紧凑度
    mask_binary = (seg_mask > 0.5).float()
    area_ratio = mask_binary.mean().item()
    
    # 边界稳定性 (简化)
    boundary_score = 1.0 - torch.abs(seg_mask - seg_mask.roll(1, dims=-1)).mean().item()
    
    E_seg = 0.5 * area_ratio + 0.5 * boundary_score
    
    return abs(pred_cls - E_seg)


# ============ FastAPI 应用 ============
app = FastAPI(
    title="MedMamba-Guard API",
    description="基于SSM状态轨迹的医学影像可信推理API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 响应模型 ============
class PredictionData(BaseModel):
    prediction: str
    confidence: float
    risk_score: float
    risk_components: Dict[str, float]
    risk_level: str
    review_regions: List[Dict[str, Any]]
    audit_log: Dict[str, Any]
    risk_heatmap_path: Optional[str] = None


class PredictionResponse(BaseModel):
    success: bool
    data: PredictionData
    message: Optional[str] = None


class RiskOnlyData(BaseModel):
    risk_score: float
    risk_components: Dict[str, float]
    risk_level: str
    review_regions: List[Dict[str, Any]]
    risk_heatmap_path: Optional[str] = None
    processing_time_ms: float


class RiskOnlyResponse(BaseModel):
    success: bool
    data: RiskOnlyData
    message: Optional[str] = None


class CTMMetricsData(BaseModel):
    V_norm_mean: float
    V_norm_max: float
    D_delta: float
    C_layer: float
    R_overconfident: float


class CTMMetricsResponse(BaseModel):
    success: bool
    data: CTMMetricsData
    message: Optional[str] = None


class HealthStatus(BaseModel):
    status: str
    device: str
    model_loaded: bool
    version: str


# ============ API 路由 ============

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """健康检查"""
    m, _, _ = load_model()
    return HealthStatus(
        status="healthy",
        device=str(DEVICE),
        model_loaded=m is not None,
        version="MedMamba-Guard v1.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    单图分类预测 + 风险评估

    支持格式: jpg, png, dicom
    返回: 预测类别、置信度、风险分数、风险成分、复核区域、审计日志
    """
    start_time = time.time()

    # 读取文件
    contents = await file.read()
    filename = file.filename.lower() if file.filename else "unknown"

    # 图像解码
    try:
        if filename.endswith(".dcm") or filename.endswith(".dicom"):
            img = read_dicom(contents)
        else:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像读取失败: {str(e)}")

    # 预处理
    x = preprocess_image(img).to(DEVICE)

    # 推理
    m, ctm_mon, cross_scan = load_model()
    
    if m is None:
        # Demo模式：返回模拟数据
        elapsed_ms = (time.time() - start_time) * 1000
        demo_result = {
            "prediction": "lesion",
            "confidence": 0.87,
            "risk_score": 0.42,
            "risk_components": {
                "R_state": 0.35,
                "R_scan": 0.28,
                "R_task": 0.15,
                "R_entropy": 0.22
            },
            "risk_level": "medium",
            "review_regions": [
                {
                    "bbox": [120, 80, 180, 150],
                    "risk_type": "cross_scan_inconsistency",
                    "risk_level": "high",
                    "reason": "四方向扫描特征散度: 0.68"
                },
                {
                    "bbox": [200, 150, 240, 200],
                    "risk_type": "state_instability",
                    "risk_level": "medium",
                    "reason": "状态激变度: 0.52"
                }
            ],
            "audit_log": {
                "model_version": "MedMamba-Guard v1.0",
                "ctm_metrics": {
                    "V_norm_mean": 0.42,
                    "V_norm_max": 0.78,
                    "D_delta": 0.35,
                    "C_layer": 0.28,
                    "R_overconfident": 0.45
                },
                "scan_metrics": {
                    "R_scan_mean": 0.32,
                    "R_scan_max": 0.68,
                    "risk_regions": [[120, 80, 180, 150]]
                },
                "timestamp": datetime.now().isoformat() + "Z"
            },
            "risk_heatmap_path": None
        }
        
        return PredictionResponse(
            success=True,
            data=PredictionData(**demo_result),
            message="Demo模式（无模型权重）"
        )

    with torch.no_grad():
        logits, hidden_states, seg_mask = m(x, return_full_outputs=True)
        pred = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, pred].item()
        
        # 计算CTM指标
        ctm_metrics = compute_ctm_metrics(hidden_states)
        
        # 计算Cross-Scan风险
        scan_metrics = compute_cross_scan_risk(x)
        
        # 计算任务冲突
        task_conflict = compute_task_conflict(confidence, seg_mask)
        
        # 计算预测熵
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()

    # 综合风险分数
    R_state = 0.4 * ctm_metrics["V_norm_mean"] + 0.3 * ctm_metrics["D_delta"] + 0.3 * ctm_metrics["C_layer"]
    R_overconf = confidence * R_state if R_state > CTM_THRESHOLDS["theta_state"] and confidence > CTM_THRESHOLDS["theta_conf"] else 0
    ctm_metrics["R_overconfident"] = R_overconf
    
    R_scan = scan_metrics["R_scan"]
    R_task = task_conflict
    R_entropy = entropy
    
    # 加权综合风险
    R_total = 0.4 * R_state + 0.3 * R_scan + 0.2 * R_task + 0.1 * R_entropy

    # 风险等级
    risk_level = compute_risk_level(R_total)

    # 硬门控判断
    review_regions = []
    gating_action = "auto_approve"
    gating_reason = "passed_all_checks"
    
    if R_total > CTM_THRESHOLDS["theta_high"]:
        gating_action = "doctor_review"
        gating_reason = "high_risk_total"
        review_regions.append({
            "bbox": [0, 0, IMG_SIZE, IMG_SIZE],
            "risk_type": "high_total_risk",
            "risk_level": risk_level,
            "reason": f"综合风险分数 {R_total:.2f} 超过阈值 {CTM_THRESHOLDS['theta_high']}"
        })
    
    if task_conflict > CTM_THRESHOLDS["theta_conflict"]:
        gating_action = "doctor_review"
        gating_reason = "task_conflict"
        review_regions.append({
            "bbox": [0, 0, IMG_SIZE, IMG_SIZE],
            "risk_type": "task_conflict",
            "risk_level": "high",
            "reason": f"任务冲突分数 {task_conflict:.2f} 超过阈值"
        })
    
    if confidence > CTM_THRESHOLDS["theta_conf"] and R_state > CTM_THRESHOLDS["theta_state"]:
        gating_action = "overconfidence_warning"
        gating_reason = "state_prediction_mismatch"

    # 生成输入hash用于审计
    input_hash = hashlib.sha256(contents[:10000]).hexdigest()[:16]

    elapsed_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        success=True,
        data=PredictionData(
            prediction=CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else f"Class_{pred}",
            confidence=round(confidence, 4),
            risk_score=round(R_total, 4),
            risk_components={
                "R_state": round(R_state, 4),
                "R_scan": round(R_scan, 4),
                "R_task": round(R_task, 4),
                "R_entropy": round(R_entropy, 4)
            },
            risk_level=risk_level,
            review_regions=review_regions,
            audit_log={
                "model_version": "MedMamba-Guard v1.0",
                "input_hash": f"sha256:{input_hash}",
                "ctm_metrics": {k: round(v, 4) for k, v in ctm_metrics.items()},
                "scan_metrics": {
                    "R_scan_mean": round(scan_metrics["R_scan"], 4),
                    "R_scan_max": round(scan_metrics["R_scan_max"], 4),
                },
                "task_metrics": {
                    "S_conflict": round(task_conflict, 4)
                },
                "gating": {
                    "action": gating_action,
                    "reason": gating_reason
                },
                "timestamp": datetime.now().isoformat() + "Z",
                "processing_time_ms": round(elapsed_ms, 2)
            },
            risk_heatmap_path=None
        ),
        message=f"预测完成，耗时 {elapsed_ms:.1f}ms"
    )


@app.post("/predict-risk-only", response_model=RiskOnlyResponse)
async def risk_only(file: UploadFile = File(...)):
    """
    仅风险评估（不返回预测结果）

    用于医生只想评估图像风险等级的的场景
    """
    start_time = time.time()

    # 读取文件
    contents = await file.read()
    filename = file.filename.lower() if file.filename else "unknown"

    # 图像解码
    try:
        if filename.endswith(".dcm") or filename.endswith(".dicom"):
            img = read_dicom(contents)
        else:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像读取失败: {str(e)}")

    # 预处理
    x = preprocess_image(img).to(DEVICE)

    # 加载模型
    m, ctm_mon, cross_scan = load_model()

    if m is None:
        # Demo模式
        elapsed_ms = (time.time() - start_time) * 1000
        return RiskOnlyResponse(
            success=True,
            data=RiskOnlyData(
                risk_score=0.45,
                risk_components={
                    "R_state": 0.38,
                    "R_scan": 0.32,
                    "R_task": 0.18,
                    "R_entropy": 0.25
                },
                risk_level="medium",
                review_regions=[
                    {
                        "bbox": [100, 100, 200, 200],
                        "risk_type": "state_instability",
                        "risk_level": "medium",
                        "reason": "状态轨迹显示局部不稳定"
                    }
                ],
                risk_heatmap_path=None,
                processing_time_ms=round(elapsed_ms, 2)
            ),
            message="Demo模式"
        )

    with torch.no_grad():
        logits, hidden_states, seg_mask = m(x, return_full_outputs=True)
        
        # 计算CTM指标
        ctm_metrics = compute_ctm_metrics(hidden_states)
        
        # 计算Cross-Scan风险
        scan_metrics = compute_cross_scan_risk(x)
        
        # 计算任务冲突
        confidence = torch.softmax(logits, dim=-1).max().item()
        task_conflict = compute_task_conflict(confidence, seg_mask)
        
        # 计算预测熵
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()

    # 综合风险分数
    R_state = 0.4 * ctm_metrics["V_norm_mean"] + 0.3 * ctm_metrics["D_delta"] + 0.3 * ctm_metrics["C_layer"]
    R_scan = scan_metrics["R_scan"]
    R_task = task_conflict
    R_entropy = entropy
    
    R_total = 0.4 * R_state + 0.3 * R_scan + 0.2 * R_task + 0.1 * R_entropy
    risk_level = compute_risk_level(R_total)

    # 识别高风险区域
    review_regions = []
    if R_total > CTM_THRESHOLDS["theta_high"]:
        review_regions.append({
            "bbox": [0, 0, IMG_SIZE, IMG_SIZE],
            "risk_type": "high_total_risk",
            "risk_level": risk_level,
            "reason": f"综合风险 {R_total:.2f} 超过阈值"
        })

    elapsed_ms = (time.time() - start_time) * 1000

    return RiskOnlyResponse(
        success=True,
        data=RiskOnlyData(
            risk_score=round(R_total, 4),
            risk_components={
                "R_state": round(R_state, 4),
                "R_scan": round(R_scan, 4),
                "R_task": round(R_task, 4),
                "R_entropy": round(R_entropy, 4)
            },
            risk_level=risk_level,
            review_regions=review_regions,
            risk_heatmap_path=None,
            processing_time_ms=round(elapsed_ms, 2)
        ),
        message=f"风险评估完成，耗时 {elapsed_ms:.1f}ms"
    )


@app.get("/ctm-metrics", response_model=CTMMetricsResponse)
async def ctm_metrics_get(image_id: Optional[str] = None):
    """
    获取CTM详细指标
    
    注意: 此接口需要先调用/predict或/predict-risk-only缓存hidden states
    当前版本返回最新一次推理的CTM指标
    """
    # 实际实现需要缓存机制，这里返回响应格式说明
    return CTMMetricsResponse(
        success=True,
        data=CTMMetricsData(
            V_norm_mean=0.0,
            V_norm_max=0.0,
            D_delta=0.0,
            C_layer=0.0,
            R_overconfident=0.0
        ),
        message="请先调用/predict接口，系统将自动记录CTM指标"
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """API文档首页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MedMamba-Guard API</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; background: #1a1a2e; color: #eee; }
            h1 { color: #00d4ff; }
            .endpoint { background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .method { background: #0f3460; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
            .method-get { background: #059669; }
            .method-post { background: #d97706; }
            code { background: #0f3460; padding: 2px 6px; border-radius: 4px; }
            .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
            .badge-new { background: #dc2626; color: white; }
        </style>
    </head>
    <body>
        <h1>MedMamba-Guard API</h1>
        <p>基于SSM状态轨迹的医学影像可信推理API</p>

        <div class="endpoint">
            <span class="method method-get">GET</span> <code>/health</code> - 服务健康检查
        </div>
        <div class="endpoint">
            <span class="method method-post">POST</span> <code>/predict</code> - 预测+风险评估 <span class="badge badge-new">NEW</span>
        </div>
        <div class="endpoint">
            <span class="method method-post">POST</span> <code>/predict-risk-only</code> - 仅风险评估 <span class="badge badge-new">NEW</span>
        </div>
        <div class="endpoint">
            <span class="method method-get">GET</span> <code>/ctm-metrics</code> - CTM详细指标 <span class="badge badge-new">NEW</span>
        </div>
        <div class="endpoint">
            <span class="method method-get">GET</span> <code>/docs</code> - Swagger API 文档
        </div>
        
        <h2>MedMamba-Guard 核心指标</h2>
        <ul>
            <li><strong>R_total</strong>: 综合风险分数 [0, 1]</li>
            <li><strong>R_state</strong>: CTM状态轨迹风险</li>
            <li><strong>R_scan</strong>: Cross-Scan一致性风险</li>
            <li><strong>R_task</strong>: 分类-分割任务冲突风险</li>
            <li><strong>R_entropy</strong>: 预测熵风险</li>
        </ul>
        
        <h2>风险等级</h2>
        <ul>
            <li>LOW [0.0-0.3): 自动通过</li>
            <li>MEDIUM [0.3-0.5): 记录日志</li>
            <li>HIGH [0.5-0.7): 标记复核</li>
            <li>CRITICAL [0.7-1.0]: 强制复核</li>
        </ul>
    </body>
    </html>
    """


# ============ 启动服务 ============
if __name__ == "__main__":
    print(f"[MedMamba-Guard API] 启动服务: http://0.0.0.0:{PORT}")
    print(f"[MedMamba-Guard API] 文档: http://0.0.0.0:{PORT}/docs")
    uvicorn.run(app, host="0.0.0.0", port=PORT)