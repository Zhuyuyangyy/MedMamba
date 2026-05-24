"""
MedMamba Benchmark - 模型性能测试

测试内容:
1. 参数量 (Parameters)
2. FLOPs (Floating Point Operations)
3. 推理延迟 (Inference Latency)
4. 内存占用 (Memory Usage)

支持模型:
- MedMamba V1 (原始Mamba)
- MedMamba V2 (CNN-SSM双分支 + CTM)
- MedMamba V3 (VMamba + HoME-MoE + CTM)

使用方法:
    python benchmark.py --model v2 --input 224
    python benchmark.py --model v3 --compare

作者: MedMamba Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

# 尝试导入thop用于FLOPs计算
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. FLOPs calculation will be skipped.")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.medmamba import MedMambaV2, MedMambaV3, create_medmamba
from src.models.selective_state_space import MambaBlock


# =============================================================================
# 工具函数
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str = "cuda",
    num_warmup: int = 10,
    num_runs: int = 100,
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    测量推理延迟
    
    Returns:
        {
            "mean_ms": 平均延迟,
            "std_ms": 标准差,
            "min_ms": 最小延迟,
            "max_ms": 最大延迟,
            "throughput_fps": 吞吐量
        }
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 测量
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    
    latencies = torch.tensor(latencies)
    
    return {
        "mean_ms": latencies.mean().item(),
        "std_ms": latencies.std().item(),
        "min_ms": latencies.min().item(),
        "max_ms": latencies.max().item(),
        "throughput_fps": 1000.0 / latencies.mean().item(),
    }


def estimate_flops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> Optional[float]:
    """
    估算FLOPs
    
    尝试使用thop库，如果不可用则使用近似估算
    """
    if HAS_THOP:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_input = torch.randn(input_shape, device=device)
        
        try:
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            return flops / 1e9  # GFLOPs
        except Exception as e:
            print(f"FLOPs profile failed: {e}")
            return None
    else:
        # 近似估算 (基于参数量)
        params = count_parameters(model)
        # 假设每个参数一次乘加操作
        return params * 2 / 1e9


def measure_memory(model: nn.Module, input_shape: Tuple[int, int, int, int], device: str = "cuda") -> Dict[str, float]:
    """测量内存占用"""
    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = model.to(device)
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
    }


def print_benchmark_results(results: Dict):
    """打印基准测试结果"""
    print("\n" + "=" * 60)
    print("                   MedMamba Benchmark Results")
    print("=" * 60)
    print(f"Model: {results['model_name']}")
    print(f"Input Shape: {results['input_shape']}")
    print("-" * 60)
    
    print("\n[Parameters]")
    print(f"  Total:        {results['total_params']:,}")
    print(f"  Trainable:    {results['trainable_params']:,}")
    
    if results.get('flops_gflops') is not None:
        print(f"\n[FLOPs]")
        print(f"  GFLOPs:       {results['flops_gflops']:.2f}")
    
    print(f"\n[Inference Latency] ({results.get('num_runs', 100)} runs)")
    print(f"  Mean:         {results['latency_mean_ms']:.2f} ms")
    print(f"  Std:          {results['latency_std_ms']:.2f} ms")
    print(f"  Min:          {results['latency_min_ms']:.2f} ms")
    print(f"  Max:          {results['latency_max_ms']:.2f} ms")
    print(f"  Throughput:   {results['throughput_fps']:.2f} FPS")
    
    if results.get('memory_allocated_mb', 0) > 0:
        print(f"\n[Memory]")
        print(f"  Allocated:    {results['memory_allocated_mb']:.2f} MB")
        print(f"  Reserved:     {results['memory_reserved_mb']:.2f} MB")
    
    print("=" * 60)


# =============================================================================
# 基准测试函数
# =============================================================================

def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_shape: Tuple[int, int, int, int],
    device: str = "cuda",
    num_runs: int = 100,
    use_amp: bool = False,
) -> Dict:
    """
    运行完整的基准测试
    """
    print(f"\n{'='*40}")
    print(f"Benchmarking {model_name}...")
    print(f"{'='*40}")
    print(f"Device: {device}")
    print(f"Input Shape: {input_shape}")
    print(f"Use AMP: {use_amp}")
    
    # 参数统计
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # FLOPs
    flops_gflops = estimate_flops(model, input_shape)
    if flops_gflops is not None:
        print(f"GFLOPs: {flops_gflops:.2f}")
    
    # 推理延迟
    print(f"\nMeasuring latency ({num_runs} runs)...")
    latency_results = measure_inference_latency(
        model, input_shape, device=device,
        num_warmup=10, num_runs=num_runs, use_amp=use_amp
    )
    
    # 内存
    if device == "cuda" and torch.cuda.is_available():
        print("Measuring memory...")
        memory_results = measure_memory(model, input_shape, device)
    else:
        memory_results = {"allocated_mb": 0, "reserved_mb": 0}
    
    # 汇总结果
    results = {
        "model_name": model_name,
        "input_shape": input_shape,
        "device": device,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops_gflops": flops_gflops,
        "num_runs": num_runs,
        "latency_mean_ms": latency_results["mean_ms"],
        "latency_std_ms": latency_results["std_ms"],
        "latency_min_ms": latency_results["min_ms"],
        "latency_max_ms": latency_results["max_ms"],
        "throughput_fps": latency_results["throughput_fps"],
        "memory_allocated_mb": memory_results["allocated_mb"],
        "memory_reserved_mb": memory_results["reserved_mb"],
    }
    
    print_benchmark_results(results)
    
    return results


def compare_models(
    input_shape: Tuple[int, int, int, int],
    device: str = "cuda",
    **kwargs
) -> List[Dict]:
    """
    比较多个模型版本的性能
    """
    print("\n" + "#" * 60)
    print("#            Model Comparison")
    print("#" * 60)
    
    results = []
    
    # V1
    try:
        model_v1 = create_medmamba(version="v1", d_model=384, n_layers=12)
        r = benchmark_model(model_v1, "MedMamba V1", input_shape, device, **kwargs)
        results.append(r)
    except Exception as e:
        print(f"V1 benchmark failed: {e}")
    
    # V2
    try:
        model_v2 = create_medmamba(version="v2", d_model=384, n_layers=12)
        r = benchmark_model(model_v2, "MedMamba V2", input_shape, device, **kwargs)
        results.append(r)
    except Exception as e:
        print(f"V2 benchmark failed: {e}")
    
    # V3
    try:
        model_v3 = create_medmamba(version="v3", d_model=384, n_layers=12)
        r = benchmark_model(model_v3, "MedMamba V3", input_shape, device, **kwargs)
        results.append(r)
    except Exception as e:
        print(f"V3 benchmark failed: {e}")
    
    # 打印对比表
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("                         Model Comparison Summary")
        print("=" * 80)
        print(f"{'Model':<20} {'Params(M)':<12} {'GFLOPs':<10} {'Lat(ms)':<12} {'FPS':<10}")
        print("-" * 80)
        
        for r in results:
            params_m = r['total_params'] / 1e6
            flops = f"{r['flops_gflops']:.1f}" if r['flops_gflops'] else "N/A"
            print(f"{r['model_name']:<20} {params_m:<12.2f} {flops:<10} {r['latency_mean_ms']:<12.2f} {r['throughput_fps']:<10.2f}")
        
        print("=" * 80)
    
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedMamba Benchmark")
    parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2", "v3", "compare"],
                        help="Model version to benchmark")
    parser.add_argument("--input", type=int, default=224,
                        help="Input image size (H=W)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models")
    
    args = parser.parse_args()
    
    input_size = args.input
    input_shape = (args.batch, 3, input_size, input_size)
    
    if args.compare or args.model == "compare":
        # 比较模式
        compare_models(
            input_shape=input_shape,
            device=args.device,
            num_runs=args.runs,
            use_amp=args.amp,
        )
    else:
        # 单模型基准测试
        model = create_medmamba(version=args.model, d_model=384, n_layers=12)
        
        results = benchmark_model(
            model=model,
            model_name=f"MedMamba {args.model.upper()}",
            input_shape=input_shape,
            device=args.device,
            num_runs=args.runs,
            use_amp=args.amp,
        )
        
        # 保存结果
        import json
        output_file = f"benchmark_{args.model}_{input_size}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()