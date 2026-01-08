#!/usr/bin/env python3
"""
Log analysis script for chess training.

Reads JSONL log files and produces:
- Summary statistics
- Bottleneck analysis
- Optional plots
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


def load_log_entries(log_path: str) -> List[Dict[str, Any]]:
    """Load all entries from JSONL log file."""
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def filter_train_entries(entries: List[Dict]) -> List[Dict]:
    """Filter to training entries only."""
    return [e for e in entries if e.get('type') == 'TrainLogEntry']


def filter_eval_entries(entries: List[Dict]) -> List[Dict]:
    """Filter to evaluation entries only."""
    return [e for e in entries if e.get('type') == 'EvalLogEntry']


@dataclass
class TrainingSummary:
    """Summary of training run."""
    total_steps: int
    total_epochs: int
    duration_minutes: float
    
    # Loss
    final_loss: float
    best_loss: float
    
    # Throughput
    avg_throughput: float
    peak_throughput: float
    
    # Timing breakdown (percentage)
    data_time_pct: float
    compute_time_pct: float
    backward_time_pct: float
    
    # GPU
    avg_gpu_util: float
    peak_memory_mb: float
    
    # Stability
    overflow_count: int
    avg_grad_norm: float


def compute_summary(entries: List[Dict]) -> Optional[TrainingSummary]:
    """Compute summary statistics from training entries."""
    train_entries = filter_train_entries(entries)
    
    if not train_entries:
        return None
    
    # Steps and epochs
    steps = [e['step'] for e in train_entries]
    epochs = [e.get('epoch', 0) for e in train_entries]
    
    # Duration
    start_time = datetime.fromisoformat(train_entries[0]['timestamp'])
    end_time = datetime.fromisoformat(train_entries[-1]['timestamp'])
    duration = (end_time - start_time).total_seconds() / 60
    
    # Loss
    losses = [e['total_loss'] for e in train_entries if e.get('total_loss')]
    
    # Throughput
    throughputs = [e['positions_per_sec'] for e in train_entries if e.get('positions_per_sec', 0) > 0]
    
    # Timing
    step_times = [e.get('step_time_ms', 0) for e in train_entries]
    data_times = [e.get('data_time_ms', 0) for e in train_entries]
    compute_times = [e.get('compute_time_ms', 0) for e in train_entries]
    backward_times = [e.get('backward_time_ms', 0) for e in train_entries]
    
    total_step_time = sum(step_times) if step_times else 1
    
    # GPU
    gpu_utils = [e.get('gpu_util_percent', 0) for e in train_entries]
    gpu_mems = [e.get('gpu_mem_peak_mb', 0) for e in train_entries]
    
    # Stability
    overflows = sum(1 for e in train_entries if e.get('amp_overflow', False))
    grad_norms = [e.get('grad_norm', 0) for e in train_entries if e.get('grad_norm')]
    
    return TrainingSummary(
        total_steps=max(steps) if steps else 0,
        total_epochs=max(epochs) if epochs else 0,
        duration_minutes=duration,
        final_loss=losses[-1] if losses else 0,
        best_loss=min(losses) if losses else 0,
        avg_throughput=sum(throughputs) / len(throughputs) if throughputs else 0,
        peak_throughput=max(throughputs) if throughputs else 0,
        data_time_pct=100 * sum(data_times) / total_step_time if total_step_time else 0,
        compute_time_pct=100 * sum(compute_times) / total_step_time if total_step_time else 0,
        backward_time_pct=100 * sum(backward_times) / total_step_time if total_step_time else 0,
        avg_gpu_util=sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
        peak_memory_mb=max(gpu_mems) if gpu_mems else 0,
        overflow_count=overflows,
        avg_grad_norm=sum(grad_norms) / len(grad_norms) if grad_norms else 0,
    )


def print_summary(summary: TrainingSummary, eval_entries: List[Dict]):
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Progress:")
    print(f"   Steps: {summary.total_steps:,}")
    print(f"   Epochs: {summary.total_epochs}")
    print(f"   Duration: {summary.duration_minutes:.1f} min")
    
    print(f"\n📉 Loss:")
    print(f"   Final: {summary.final_loss:.4f}")
    print(f"   Best: {summary.best_loss:.4f}")
    
    print(f"\n⚡ Throughput:")
    print(f"   Average: {summary.avg_throughput:,.0f} pos/sec")
    print(f"   Peak: {summary.peak_throughput:,.0f} pos/sec")
    
    print(f"\n⏱️ Time Breakdown:")
    other_pct = max(0, 100 - summary.data_time_pct - summary.compute_time_pct - summary.backward_time_pct)
    print(f"   Data loading: {summary.data_time_pct:.1f}%")
    print(f"   Forward pass: {summary.compute_time_pct:.1f}%")
    print(f"   Backward pass: {summary.backward_time_pct:.1f}%")
    print(f"   Other: {other_pct:.1f}%")
    
    # Bottleneck analysis
    print(f"\n🔍 Bottleneck Analysis:")
    if summary.data_time_pct > 40:
        print("   ⚠️  DATA LOADING is the bottleneck!")
        print("      → Try: more workers, faster storage, larger prefetch")
    elif summary.compute_time_pct + summary.backward_time_pct > 80:
        print("   ✓ GPU-bound (optimal)")
    else:
        print("   ⚠️  Significant overhead detected")
        print("      → Check: batch size, model complexity, I/O")
    
    print(f"\n🖥️ GPU:")
    print(f"   Utilization: {summary.avg_gpu_util:.1f}%")
    print(f"   Peak Memory: {summary.peak_memory_mb:.0f} MB")
    
    print(f"\n🔧 Stability:")
    print(f"   AMP Overflows: {summary.overflow_count}")
    print(f"   Avg Grad Norm: {summary.avg_grad_norm:.2f}")
    
    # Evaluation results
    if eval_entries:
        print(f"\n📈 Evaluation Results:")
        for entry in eval_entries:
            if entry.get('eval_type') == 'offline':
                print(f"   Step {entry['step']}: Policy Acc={entry['policy_accuracy']:.2%}, Value MSE={entry['value_mse']:.4f}")
            elif entry.get('eval_type') == 'elo':
                print(f"   Step {entry['step']}: Elo={entry['elo_estimate']:.0f} (Score={entry['score']:.1%})")
    
    print("\n" + "=" * 60)


def plot_training(entries: List[Dict], output_dir: str = None):
    """Generate training plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    train_entries = filter_train_entries(entries)
    eval_entries = filter_eval_entries(entries)
    
    if not train_entries:
        return
    
    steps = [e['step'] for e in train_entries]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=14)
    
    # Loss plot
    ax = axes[0, 0]
    policy_loss = [e.get('policy_loss', 0) for e in train_entries]
    value_loss = [e.get('value_loss', 0) for e in train_entries]
    total_loss = [e.get('total_loss', 0) for e in train_entries]
    
    ax.plot(steps, policy_loss, label='Policy', alpha=0.7)
    ax.plot(steps, value_loss, label='Value', alpha=0.7)
    ax.plot(steps, total_loss, label='Total', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Throughput plot
    ax = axes[0, 1]
    throughput = [e.get('positions_per_sec', 0) for e in train_entries]
    ax.plot(steps, throughput, color='green', alpha=0.7)
    ax.axhline(y=sum(throughput)/len(throughput), color='green', linestyle='--', label='Average')
    ax.set_xlabel('Step')
    ax.set_ylabel('Positions/sec')
    ax.set_title('Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Timing breakdown (stacked area)
    ax = axes[1, 0]
    data_time = [e.get('data_time_ms', 0) for e in train_entries]
    compute_time = [e.get('compute_time_ms', 0) for e in train_entries]
    backward_time = [e.get('backward_time_ms', 0) for e in train_entries]
    
    ax.stackplot(steps, data_time, compute_time, backward_time,
                 labels=['Data', 'Forward', 'Backward'],
                 alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Time Breakdown')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # GPU memory
    ax = axes[1, 1]
    gpu_mem = [e.get('gpu_mem_mb', 0) for e in train_entries]
    ax.plot(steps, gpu_mem, color='purple', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('GPU Memory Usage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_plots.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
    else:
        plt.savefig('training_plots.png', dpi=150, bbox_inches='tight')
        print("Plots saved to: training_plots.png")
    
    plt.close()
    
    # Elo plot if available
    elo_entries = [e for e in eval_entries if e.get('eval_type') == 'elo']
    if elo_entries:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        elo_steps = [e['step'] for e in elo_entries]
        elos = [e['elo_estimate'] for e in elo_entries]
        
        ax.plot(elo_steps, elos, 'o-', color='blue', linewidth=2, markersize=8)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Estimated Elo')
        ax.set_title('Elo Progression')
        ax.grid(True, alpha=0.3)
        
        if output_dir:
            output_path = os.path.join(output_dir, 'elo_progression.png')
        else:
            output_path = 'elo_progression.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Elo plot saved to: {output_path}")
        plt.close()


def export_csv(entries: List[Dict], output_path: str):
    """Export training metrics to CSV."""
    import csv
    
    train_entries = filter_train_entries(entries)
    if not train_entries:
        print("No training entries to export")
        return
    
    keys = list(train_entries[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(train_entries)
    
    print(f"CSV exported to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze training logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("log_file", type=str,
                        help="Path to JSONL log file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="Export to CSV file")
    parser.add_argument("--json-summary", action="store_true",
                        help="Output summary as JSON")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)
    
    # Load entries
    entries = load_log_entries(args.log_file)
    print(f"Loaded {len(entries)} log entries")
    
    # Compute summary
    summary = compute_summary(entries)
    eval_entries = filter_eval_entries(entries)
    
    if summary:
        if args.json_summary:
            import dataclasses
            print(json.dumps(dataclasses.asdict(summary), indent=2))
        else:
            print_summary(summary, eval_entries)
    
    # Optional outputs
    if args.plot:
        plot_training(entries, args.output_dir)
    
    if args.export_csv:
        export_csv(entries, args.export_csv)


if __name__ == "__main__":
    main()
