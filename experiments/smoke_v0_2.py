
"""
MedMamba-Guard V0.2 Smoke Experiment
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.medmamba_guard import MedMambaGuard
from src.models.ssm_config import MedMambaGuardConfig


class SyntheticDataGenerator:
    def __init__(self, img_size: int = 32):
        self.img_size = img_size

    def generate_clean_sample(self, class_idx: int = 0):
        img = torch.randn(1, 3, self.img_size, self.img_size) * 0.2 + 0.5
        img.clamp_(0, 1)

        if class_idx == 1:
            center = self.img_size // 2
            y, x = torch.meshgrid(
                torch.arange(self.img_size),
                torch.arange(self.img_size),
                indexing='ij'
            )
            mask = (x - center) ** 2 + (y - center) ** 2 < (self.img_size // 6) ** 2
            img[:, 0, mask] += 0.4
        return img

    def generate_blur_sample(self, class_idx: int = 0):
        img = self.generate_clean_sample(class_idx)
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        for c in range(3):
            img[:, c:c+1] = F.conv2d(
                F.pad(img[:, c:c+1], (1, 1, 1, 1), mode='reflect'),
                kernel,
                padding='valid'
            )
        return img

    def generate_noise_sample(self, class_idx: int = 0, noise_level: float = 0.3):
        img = self.generate_clean_sample(class_idx)
        img += torch.randn_like(img) * noise_level
        img.clamp_(0, 1)
        return img

    def generate_conflict_sample(self, class_idx: int = 0):
        return self.generate_clean_sample(1 - class_idx)


@dataclass
class SmokeConfig:
    mode: str = "synthetic"
    max_samples: int = 20
    img_size: int = 32
    batch_size: int = 1
    output_dir: str = "./smoke_results"
    dataset: Optional[str] = None
    data_root: Optional[str] = None
    d_model: int = 32
    n_layers: int = 2
    num_classes: int = 2
    device: str = "cpu"


class SmokeExperimentRunner:
    def __init__(self, config: SmokeConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._init_model()
        self.model.eval()

    def _init_model(self):
        model_config = MedMambaGuardConfig(
            d_model=self.config.d_model,
            d_state=4,
            d_conv=3,
            expand=2,
            num_classes=self.config.num_classes,
            dropout=0.0,
            img_size=self.config.img_size,
            patch_size=4,
            n_layers=self.config.n_layers,
        )
        return MedMambaGuard(model_config).to(self.config.device)

    @torch.no_grad()
    def run_synthetic(self):
        print("\n" + "="*70)
        print("Running Synthetic Smoke Experiment")
        print("="*70)

        data_gen = SyntheticDataGenerator(self.config.img_size)
        all_conditions = ['clean', 'blur', 'noise', 'conflict']
        samples_per_cond = self.config.max_samples // len(all_conditions)
        n_total = samples_per_cond * len(all_conditions)
        print(f"\nGenerating {n_total} synthetic samples...")

        results_data = []
        for cond_idx, condition in enumerate(all_conditions):
            print(f"\nProcessing condition: {condition}")
            for sample_idx in range(samples_per_cond):
                label = torch.tensor(np.random.randint(0, 2))
                if condition == 'clean':
                    img = data_gen.generate_clean_sample(label.item())
                elif condition == 'blur':
                    img = data_gen.generate_blur_sample(label.item())
                elif condition == 'noise':
                    img = data_gen.generate_noise_sample(label.item())
                elif condition == 'conflict':
                    img = data_gen.generate_conflict_sample(label.item())
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                img = img.to(self.config.device)
                outputs = self.model(
                    img,
                    mode='eval',
                    return_risk=True,
                    return_audit=True,
                )

                # Get pred and conf from outputs
                pred = torch.tensor(1 if outputs.get('prediction') == 'lesion' else 0)
                conf = torch.tensor(outputs.get('confidence', 0.5))

                risk_comp = outputs.get('risk_components', {})

                r_state = risk_comp.get('R_state', torch.tensor(0.0))
                r_scan = risk_comp.get('R_scan', torch.tensor(0.0))
                r_task = risk_comp.get('R_task', torch.tensor(0.0))
                r_entropy = risk_comp.get('R_entropy', torch.tensor(0.0))
                r_total = outputs.get('risk_score', 0.0)

                if isinstance(r_state, torch.Tensor):
                    r_state = r_state.mean().item()
                if isinstance(r_scan, torch.Tensor):
                    r_scan = r_scan.mean().item()
                if isinstance(r_task, torch.Tensor):
                    r_task = r_task.mean().item()
                if isinstance(r_entropy, torch.Tensor):
                    r_entropy = r_entropy.mean().item()
                if isinstance(r_total, torch.Tensor):
                    r_total = r_total.mean().item()

                gate_action = outputs.get('action', 'PASS')

                results_data.append({
                    'sample_id': f"{condition}_{sample_idx:03d}",
                    'condition': condition,
                    'pred': int(pred.item()),
                    'label': int(label.item()),
                    'correct': int(pred.item() == label.item()),
                    'confidence': float(conf.item()),
                    'r_state': float(r_state),
                    'r_scan': float(r_scan),
                    'r_task': float(r_task),
                    'r_entropy': float(r_entropy),
                    'r_total': float(r_total),
                    'gate_action': gate_action,
                })

                if sample_idx % 5 == 0 or sample_idx == samples_per_cond - 1:
                    print(f"  [{sample_idx+1}/{samples_per_cond}] {condition}_{sample_idx:03d}: pred={pred.item()}, label={label.item()}, R_total={r_total:.3f}, action={gate_action}")

        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "smoke_results.csv", index=False)
        df.to_json(self.output_dir / "smoke_results.json", orient='records', indent=2, force_ascii=False)
        summary = self._generate_summary(df)

        log_path = self.output_dir / "run_log.txt"
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("MedMamba-Guard V0.2 Synthetic Smoke Experiment Log\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image size: {self.config.img_size}\n")
            f.write(f"d_model: {self.config.d_model}, n_layers: {self.config.n_layers}\n")
            f.write(f"Total samples: {len(df)}\n")

        print(f"\nDone! Results saved to: {self.output_dir}")
        return {
            'results_data': results_data,
            'summary': summary,
        }

    def _generate_summary(self, df: pd.DataFrame):
        summary = {}
        summary['by_condition'] = {}
        for cond in df['condition'].unique():
            cond_df = df[df['condition'] == cond]
            summary['by_condition'][cond] = {
                'count': int(len(cond_df)),
                'r_total_mean': float(cond_df['r_total'].mean()),
                'r_state_mean': float(cond_df['r_state'].mean()),
                'r_scan_mean': float(cond_df['r_scan'].mean()),
                'r_task_mean': float(cond_df['r_task'].mean()),
                'confidence_mean': float(cond_df['confidence'].mean()),
                'accuracy': float(cond_df['correct'].mean()),
                'review_rate': float((cond_df['gate_action'] != 'PASS').mean()),
            }

        by_cond = summary['by_condition']
        clean_r = by_cond.get('clean', {}).get('r_total_mean', 999)
        blur_r = by_cond.get('blur', {}).get('r_total_mean', -1)
        noise_r = by_cond.get('noise', {}).get('r_total_mean', -1)
        conflict_r = by_cond.get('conflict', {}).get('r_total_mean', -1)

        summary['trends'] = {
            'clean_lowest_risk': clean_r < min(blur_r, noise_r, conflict_r),
            'blur_or_noise_higher_than_clean': max(blur_r, noise_r) > clean_r,
            'conflict_higher_r_task': by_cond.get('conflict', {}).get('r_task_mean', 0) > by_cond.get('clean', {}).get('r_task_mean', 999),
        }

        all_r = df[['r_state', 'r_scan', 'r_task', 'r_entropy', 'r_total']].values
        summary['no_nan_inf'] = bool(np.all(np.isfinite(all_r)))

        valid_actions = {'PASS', 'REVIEW', 'doctor_review', 'overconfidence_warning'}
        summary['valid_gate_actions'] = bool(set(df['gate_action'].unique()).issubset(valid_actions))

        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="MedMamba-Guard V0.2 Smoke Experiment")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./smoke_results")
    parser.add_argument("--dataset", type=str, default=None, choices=["isic2018", "medmnist"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    config = SmokeConfig(
        mode=args.mode,
        max_samples=args.max_samples,
        img_size=args.image_size,
        output_dir=args.output_dir,
        dataset=args.dataset,
        data_root=args.data_root,
        d_model=args.d_model,
        n_layers=args.n_layers,
        device=args.device,
    )

    runner = SmokeExperimentRunner(config)

    if args.mode == "synthetic":
        results = runner.run_synthetic()
    else:
        raise NotImplementedError("Real mode not implemented")

    print("\n" + "="*70)
    print("Smoke Experiment Summary")
    print("="*70)

    summary = results.get('summary', {})
    by_cond = summary.get('by_condition', {})

    print("\nAverage risk by condition:")
    for cond in ['clean', 'blur', 'noise', 'conflict']:
        if cond in by_cond:
            d = by_cond[cond]
            print(f"  - {cond:8s}: R_total={d['r_total_mean']:.3f} (R_state={d['r_state_mean']:.3f}, R_scan={d['r_scan_mean']:.3f}, R_task={d['r_task_mean']:.3f}) Review={d['review_rate']:.1%}")

    print("\nTrend validation:")
    for k, v in summary.get('trends', {}).items():
        status = "✅ PASS" if v else "❌"
        print(f"  {status}: {k}")

    print(f"\n✅ All risks finite: {summary.get('no_nan_inf', False)}")
    print(f"✅ Gate actions valid: {summary.get('valid_gate_actions', False)}")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

