
"""
Simple analysis script for smoke results
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(csv_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    lines = []
    lines.append("# MedMamba-Guard V0.2 Smoke Experiment Summary")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Basic stats
    lines.append("## Basic Stats")
    lines.append(f"- Total samples: {len(df)}")
    lines.append(f"- Overall accuracy: {df['correct'].mean():.2%}")
    lines.append("")
    
    # By condition
    lines.append("## By Condition")
    lines.append("")
    lines.append("| Condition | Count | Accuracy | Confidence | R_total | R_state | R_scan | R_task | Review Rate |")
    lines.append("|-----------|-------|----------|------------|---------|---------|--------|--------|-------------|")
    for cond in ['clean', 'blur', 'noise', 'conflict']:
        if cond in df['condition'].values:
            d = df[df['condition'] == cond]
            acc = d['correct'].mean()
            conf_mean = d['confidence'].mean()
            r_total = d['r_total'].mean()
            r_state = d['r_state'].mean()
            r_scan = d['r_scan'].mean()
            r_task = d['r_task'].mean()
            review_rate = (d['gate_action'] != 'PASS').mean()
            lines.append(f"| {cond} | {len(d)} | {acc:.1%} | {conf_mean:.3f} | {r_total:.3f} | {r_state:.3f} | {r_scan:.3f} | {r_task:.3f} | {review_rate:.1%} |")
    lines.append("")
    
    # Key observations
    lines.append("## Key Observations")
    lines.append("")
    by_cond = {}
    for cond in df['condition'].unique():
        d = df[df['condition'] == cond]
        by_cond[cond] = {
            'count': len(d),
            'r_total_mean': d['r_total'].mean(),
            'r_state_mean': d['r_state'].mean(),
            'r_scan_mean': d['r_scan'].mean(),
            'r_task_mean': d['r_task'].mean(),
            'confidence_mean': d['confidence'].mean(),
            'accuracy': d['correct'].mean(),
            'review_rate': (d['gate_action'] != 'PASS').mean(),
        }
    
    clean_r = by_cond.get('clean', {}).get('r_total_mean', 999)
    blur_r = by_cond.get('blur', {}).get('r_total_mean', -1)
    noise_r = by_cond.get('noise', {}).get('r_total_mean', -1)
    conflict_r = by_cond.get('conflict', {}).get('r_total_mean', -1)
    
    if clean_r < min(blur_r, noise_r, conflict_r):
        lines.append("- ✅ Clean samples have lowest R_total")
    else:
        lines.append("- ⚠️ Clean samples do NOT have lowest R_total")
    
    if max(blur_r, noise_r) > clean_r:
        lines.append("- ✅ Blur/Noise samples have higher R_total")
    else:
        lines.append("- ⚠️ Blur/Noise samples do NOT have higher R_total")
    
    if by_cond.get('conflict', {}).get('r_task_mean', -1) > by_cond.get('clean', {}).get('r_task_mean', 999):
        lines.append("- ✅ Conflict samples have higher R_task")
    else:
        lines.append("- ⚠️ Conflict samples do NOT have higher R_task")
    
    # Check NaN/Inf
    all_r = df[['r_state', 'r_scan', 'r_task', 'r_entropy', 'r_total']].values
    if np.all(np.isfinite(all_r)):
        lines.append("- ✅ All risks are finite (no NaN/Inf)")
    else:
        lines.append("- ❌ Some risks are NaN/Inf")
    
    valid_actions = {'PASS', 'REVIEW', 'doctor_review', 'overconfidence_warning'}
    if set(df['gate_action'].unique()).issubset(valid_actions):
        lines.append("- ✅ Gate actions valid")
    else:
        lines.append("- ❌ Invalid gate actions")
    
    # Write to file
    summary_path = output_dir / "risk_error_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\n" + '\n'.join(lines))
    print(f"\nReport saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze smoke experiment results")
    parser.add_argument("--input", type=str, required=True, help="Path to smoke_results.csv")
    parser.add_argument("--output", type=str, default="./", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(args.input, args.output)


if __name__ == "__main__":
    main()

