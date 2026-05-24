"""
MedMamba 医学影像分类模型 - 主入口
支持: 训练 / 推理 / 服务部署 / 幻觉热图可视化
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MedMamba - CTM-Hybrid SSM医学影像分类")
    parser.add_argument("--mode", type=str, default="info",
                        choices=["train", "eval", "serve", "info", "benchmark"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ctm-ticks", type=int, default=8)
    parser.add_argument("--hallucination-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.mode == "info":
        print("=" * 60)
        print("  MedMamba - CTM-Hybrid SSM 医学影像分类模型")
        print("=" * 60)
        print()
        print("  架构: CTM轨迹分析 + 选择性状态空间(SSM) + 目标注意力")
        print("  核心: O(n)线性复杂度, 替代Transformer的O(n²)注意力")
        print()
        print("  模型配置:")
        print(f"    - d_model:    {args.d_model}")
        print(f"    - n_layers:   {args.n_layers}")
        print(f"    - d_state:    {args.d_state}")
        print(f"    - img_size:   {args.img_size}")
        print(f"    - patch_size: {args.patch_size}")
        print(f"    - num_classes:{args.num_classes}")
        print()
        print("  创新点:")
        print("    1. SSM替代Self-Attention: O(n) vs O(n²)")
        print("    2. CTM动力学轨迹分析: 实时幻觉检测")
        print("    3. 目标注意力: 仅在高风险区触发O(n²)计算")
        print("    4. 硬件友好: 状态空间可并行扫描")
        print()
        print("  使用方式:")
        print("    python main.py --mode train  --checkpoint model.pth")
        print("    python main.py --mode eval   --checkpoint model.pth")
        print("    python main.py --mode serve  --device cpu")
        print("    python main.py --mode benchmark")
        print()
        print("  论文参考: Mamba-Linear-Time-Selective-SSM (ICLR 2024)")
        print("  CTM融合:  Continuous Thought Machine动力学分析")
        print()
        return
    
    # 延迟导入避免无torch时报错
    from src.models.medmamba import MedMamba, MedMambaClassifier
    from src.models.ssm_config import SSMConfig, MedMambaConfig
    
    device = torch.device(args.device)
    
    if args.mode == "benchmark":
        print("[Benchmark] 测试模型参数量和FLOPs...")
        
        model = MedMambaClassifier(
            img_size=args.img_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            num_heads=args.num_heads,
            ctm_ticks=args.ctm_ticks,
            hallucination_threshold=args.hallucination_threshold,
        ).to(device)
        
        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  总参数:    {total_params:,}")
        print(f"  可训练:    {trainable_params:,}")
        print(f"  模型大小:  {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        
        # 推理延迟测试
        import time
        x = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(x)
            
            # Timed
            start = time.time()
            for _ in range(10):
                _ = model(x)
            elapsed = (time.time() - start) / 10 * 1000
        
        print(f"  推理延迟:  {elapsed:.2f} ms/图")
        print(f"  吞吐量:    {1000/elapsed:.1f} 图/秒")
        print()
        print("  对比参考:")
        print(f"    - ViT-B/16:  ~87M参数, O(n²)注意力, ~15ms/图")
        print(f"    - MedMamba:   {total_params//1000000}M参数, O(n)SSM, ~{elapsed:.1f}ms/图")
        return
    
    # 加载模型
    model = MedMambaClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        num_heads=args.num_heads,
        ctm_ticks=args.ctm_ticks,
        hallucination_threshold=args.hallucination_threshold,
    ).to(device)
    
    if args.checkpoint:
        print(f"[Load] 从 {args.checkpoint} 加载权重...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    
    if args.mode == "train":
        print("[Train] 训练模式 (需配合数据加载器)...")
        print("  示例命令:")
        print("    python -c \"")
        print("      from src.models.medmamba import MedMambaClassifier")
        print("      from torch.optim import AdamW")
        print("      model = MedMambaClassifier(num_classes=2)")
        print("      x = torch.randn(4, 3, 224, 224)")
        print("      y = torch.tensor([0,1,0,1])")
        print("      criterion = nn.CrossEntropyLoss()")
        print("      optimizer = AdamW(model.parameters(), lr=1e-4)")
        print("      for epoch in range(10):")
        print("        optimizer.zero_grad()")
        print("        loss = criterion(model(x), y)")
        print("        loss.backward()")
        print("        optimizer.step()")
        print("        print(f'Epoch {epoch}, loss={loss.item():.4f}')")
        print("    \"")
        print()
        print("  可加 --checkpoint 加载预训练权重做微调")
        return
    
    if args.mode == "eval":
        print("[Eval] 评估模式...")
        x = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            logits, h_map = model(x, return_hallucination_map=True)
            pred = logits.argmax(dim=-1).item()
            h_score = h_map.mean().item() if h_map is not None else 0
        print(f"  预测类别: {pred}")
        print(f"  幻觉风险: {h_score:.4f} ({'低' if h_score < 0.3 else '中' if h_score < 0.6 else '高'})")
        return
    
    if args.mode == "serve":
        print("[Serve] 启动FastAPI服务...")
        print(f"  设备: {device}")
        print(f"  端口: 8866")
        print()
        
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import JSONResponse
        import numpy as np
        from PIL import Image
        import io
        
        app = FastAPI(title="MedMamba Server")
        
        @app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = img.resize((args.img_size, args.img_size))
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                logits, h_map = model(x, return_hallucination_map=True)
                pred = logits.argmax(dim=-1).item()
                h_score = h_map.mean().item() if h_map is not None else 0
            
            return JSONResponse({
                "prediction": pred,
                "hallucination_risk": round(h_score, 4),
                "risk_level": "low" if h_score < 0.3 else "medium" if h_score < 0.6 else "high"
            })
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8866)


if __name__ == "__main__":
    main()