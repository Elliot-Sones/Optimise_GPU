"""
Comprehensive logging utilities for chess training.

Features:
- Multi-output logging (stdout, CSV, JSONL, TensorBoard)
- Training metrics (losses, LR, grad norm, AMP)
- Throughput instrumentation (positions/sec, timing)
- System metrics (GPU/CPU utilization, memory)
- Elo evaluation results
"""

import os
import sys
import json
import time
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from collections import deque
import threading
import warnings

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================================
# Timer utilities
# ============================================================================

class Timer:
    """Simple timer for measuring durations."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0
    
    def start(self):
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def reset(self):
        self.start_time = None
        self.elapsed = 0.0


@contextmanager
def timed(name: str = None):
    """Context manager for timing code blocks."""
    timer = Timer()
    timer.start()
    yield timer
    timer.stop()


class TimerCollection:
    """Collection of named timers for multi-phase timing."""
    
    def __init__(self):
        self.timers: Dict[str, Timer] = {}
        self._active: Optional[str] = None
    
    def start(self, name: str):
        if name not in self.timers:
            self.timers[name] = Timer()
        self.timers[name].start()
        self._active = name
    
    def stop(self, name: str = None):
        name = name or self._active
        if name and name in self.timers:
            return self.timers[name].stop()
        return 0.0
    
    def get(self, name: str) -> float:
        if name in self.timers:
            return self.timers[name].elapsed
        return 0.0
    
    def get_all(self) -> Dict[str, float]:
        return {name: timer.elapsed for name, timer in self.timers.items()}
    
    def reset_all(self):
        for timer in self.timers.values():
            timer.reset()


# ============================================================================
# Moving averages and statistics
# ============================================================================

class MovingAverage:
    """Exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class WindowStats:
    """Statistics over a sliding window."""
    
    def __init__(self, window_size: int = 100):
        self.window = deque(maxlen=window_size)
    
    def update(self, x: float):
        self.window.append(x)
    
    @property
    def mean(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0
    
    @property
    def count(self) -> int:
        return len(self.window)
    
    def percentile(self, p: float) -> float:
        if not self.window:
            return 0.0
        sorted_vals = sorted(self.window)
        idx = int(p / 100 * (len(sorted_vals) - 1))
        return sorted_vals[idx]


# ============================================================================
# System metrics
# ============================================================================

@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""
    utilization_percent: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0
    memory_peak_mb: float = 0.0


@dataclass  
class CPUMetrics:
    """CPU metrics snapshot."""
    utilization_percent: float = 0.0
    memory_percent: float = 0.0


def get_gpu_metrics(device: Optional[int] = None) -> GPUMetrics:
    """Get current GPU metrics."""
    if torch is None or not torch.cuda.is_available():
        return GPUMetrics()
    
    device = device or torch.cuda.current_device()
    
    try:
        # Memory stats
        mem_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        
        # Peak memory
        mem_stats = torch.cuda.memory_stats(device)
        mem_peak = mem_stats.get('allocated_bytes.all.peak', 0) / 1024 / 1024
        
        # GPU utilization (requires pynvml or nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={device}'],
                capture_output=True, text=True, timeout=1
            )
            utilization = float(result.stdout.strip()) if result.returncode == 0 else 0.0
        except:
            utilization = 0.0
        
        return GPUMetrics(
            utilization_percent=utilization,
            memory_allocated_mb=mem_allocated,
            memory_reserved_mb=mem_reserved,
            memory_peak_mb=mem_peak,
        )
    except Exception:
        return GPUMetrics()


def get_cpu_metrics() -> CPUMetrics:
    """Get current CPU metrics."""
    if psutil is None:
        return CPUMetrics()
    
    try:
        return CPUMetrics(
            utilization_percent=psutil.cpu_percent(interval=None),
            memory_percent=psutil.virtual_memory().percent,
        )
    except Exception:
        return CPUMetrics()


# ============================================================================
# Log entries
# ============================================================================

@dataclass
class TrainLogEntry:
    """Single training log entry."""
    timestamp: str
    step: int
    epoch: int = 0
    
    # Losses
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    entropy: float = 0.0
    
    # Optimizer
    lr: float = 0.0
    grad_norm: float = 0.0
    
    # AMP
    amp_scale: float = 1.0
    amp_overflow: bool = False
    
    # Throughput
    positions_per_sec: float = 0.0
    step_time_ms: float = 0.0
    mfu_percent: float = 0.0
    tflops_per_sec: float = 0.0
    
    # Timing breakdown
    data_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    
    # System
    gpu_util_percent: float = 0.0
    gpu_mem_mb: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    cpu_util_percent: float = 0.0
    cpu_mem_percent: float = 0.0
    
    # Scaling
    gpu_count: int = 1
    effective_batch_size: int = 256


@dataclass
class EvalLogEntry:
    """Evaluation results log entry."""
    timestamp: str
    step: int
    eval_type: str  # "offline" or "elo"
    
    # Offline metrics
    policy_accuracy: float = 0.0
    value_mse: float = 0.0
    num_samples: int = 0
    
    # Elo metrics
    elo_estimate: float = 0.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score: float = 0.0
    engine_name: str = ""
    engine_settings: str = ""


# ============================================================================
# Logger
# ============================================================================

class MetricsLogger:
    """
    Comprehensive metrics logger.
    
    Outputs to:
    - stdout (pretty printed)
    - JSONL file (for analysis)
    - CSV file (for spreadsheets)
    - TensorBoard (optional)
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        log_to_stdout: bool = True,
        log_to_jsonl: bool = True,
        log_to_csv: bool = True,
        log_to_tensorboard: bool = False,
        log_to_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        config: Optional[Dict] = None,
        print_every: int = 1,
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_to_stdout = log_to_stdout
        self.log_to_jsonl = log_to_jsonl
        self.log_to_csv = log_to_csv
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_wandb = log_to_wandb and (wandb is not None)
        self.print_every = print_every
        
        # Initialize W&B
        if self.log_to_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                name=self.experiment_name,
                config=config,
            )
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # File handles
        self.jsonl_path = os.path.join(log_dir, f"{self.experiment_name}.jsonl")
        self.csv_path = os.path.join(log_dir, f"{self.experiment_name}.csv")
        
        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_headers_written = False
        
        # TensorBoard
        self._tb_writer = None
        if log_to_tensorboard and SummaryWriter is not None:
            tb_dir = os.path.join(log_dir, "tensorboard", self.experiment_name)
            self._tb_writer = SummaryWriter(tb_dir)
        
        # Statistics
        self.step_times = WindowStats(100)
        self.throughput_ema = MovingAverage(0.1)
        self.tflops_ema = MovingAverage(0.1)
        self.mfu_ema = MovingAverage(0.1)
        
        # Timers
        self.timers = TimerCollection()
        
        self._step_start_time = None
        self._positions_this_step = 0
        self._flops_per_position = 0.0
        self._gpu_peak_flops = 0.0
    
    def _get_jsonl_file(self):
        if self._jsonl_file is None and self.log_to_jsonl:
            self._jsonl_file = open(self.jsonl_path, 'a')
        return self._jsonl_file
    
    def _get_csv_file(self):
        if self._csv_file is None and self.log_to_csv:
            self._csv_file = open(self.csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file)
        return self._csv_file
    
    def _write_jsonl(self, entry: Union[TrainLogEntry, EvalLogEntry]):
        f = self._get_jsonl_file()
        if f:
            data = asdict(entry)
            data['type'] = entry.__class__.__name__
            if USE_ORJSON:
                f.write(orjson.dumps(data).decode() + '\n')
            else:
                f.write(json.dumps(data) + '\n')
            f.flush()
    
    def _write_csv(self, entry: TrainLogEntry):
        self._get_csv_file()
        if self._csv_writer:
            data = asdict(entry)
            if not self._csv_headers_written:
                self._csv_writer.writerow(data.keys())
                self._csv_headers_written = True
            self._csv_writer.writerow(data.values())
            self._csv_file.flush()
    
    def _write_tensorboard(self, entry: TrainLogEntry):
        if self._tb_writer:
            self._tb_writer.add_scalar('loss/policy', entry.policy_loss, entry.step)
            self._tb_writer.add_scalar('loss/value', entry.value_loss, entry.step)
            self._tb_writer.add_scalar('loss/total', entry.total_loss, entry.step)
            self._tb_writer.add_scalar('lr', entry.lr, entry.step)
            self._tb_writer.add_scalar('grad_norm', entry.grad_norm, entry.step)
            self._tb_writer.add_scalar('throughput/positions_per_sec', entry.positions_per_sec, entry.step)
            self._tb_writer.add_scalar('throughput/tflops', entry.tflops_per_sec, entry.step)
            self._tb_writer.add_scalar('throughput/mfu', entry.mfu_percent, entry.step)
            self._tb_writer.add_scalar('gpu/memory_mb', entry.gpu_mem_mb, entry.step)
            self._tb_writer.add_scalar('gpu/utilization', entry.gpu_util_percent, entry.step)
    
    def start_step(self, batch_size: int = 0, flops_per_position: float = 0.0, gpu_peak_flops: float = 1.0):
        """Call at the start of each training step."""
        self._step_start_time = time.perf_counter()
        self._positions_this_step = batch_size
        self._flops_per_position = flops_per_position
        self._gpu_peak_flops = gpu_peak_flops
        self.timers.reset_all()
    
    def log_train_step(
        self,
        step: int,
        epoch: int = 0,
        policy_loss: float = 0.0,
        value_loss: float = 0.0,
        total_loss: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        amp_scale: float = 1.0,
        amp_overflow: bool = False,
        entropy: float = 0.0,
        gpu_count: int = 1,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log a training step."""
        # Calculate timing
        step_time = 0.0
        if self._step_start_time:
            step_time = (time.perf_counter() - self._step_start_time) * 1000
        
        self.step_times.update(step_time)
        
        # Calculate throughput
        positions_per_sec = 0.0
        tflops = 0.0
        mfu = 0.0
        
        if step_time > 0:
            seconds = step_time / 1000
            positions_per_sec = self._positions_this_step / seconds
            
            # TFLOPS = (OPS per pos * pos per sec) / 1e12
            # 3x factor for backward pass (1 fw + 2 bw)
            total_ops = self._flops_per_position * self._positions_this_step * 3
            tflops = total_ops / seconds / 1e12
            
            # MFU = TFLOPS / Peak TFLOPS
            if self._gpu_peak_flops > 0:
                mfu = (tflops / (self._gpu_peak_flops / 1e12)) * 100
            
            self.throughput_ema.update(positions_per_sec)
            self.tflops_ema.update(tflops)
            self.mfu_ema.update(mfu)
        
        # Get system metrics
        gpu_metrics = get_gpu_metrics()
        cpu_metrics = get_cpu_metrics()
        
        # Get timing breakdown
        timings = self.timers.get_all()
        
        entry = TrainLogEntry(
            timestamp=datetime.now().isoformat(),
            step=step,
            epoch=epoch,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            lr=lr,
            grad_norm=grad_norm,
            amp_scale=amp_scale,
            amp_overflow=amp_overflow,
            positions_per_sec=positions_per_sec,
            tflops_per_sec=tflops,
            mfu_percent=mfu,
            step_time_ms=step_time,
            data_time_ms=timings.get('data', 0.0) * 1000,
            compute_time_ms=timings.get('compute', 0.0) * 1000,
            backward_time_ms=timings.get('backward', 0.0) * 1000,
            gpu_util_percent=gpu_metrics.utilization_percent,
            gpu_mem_mb=gpu_metrics.memory_allocated_mb,
            gpu_mem_peak_mb=gpu_metrics.memory_peak_mb,
            cpu_util_percent=cpu_metrics.utilization_percent,
            cpu_mem_percent=cpu_metrics.memory_percent,
            gpu_count=gpu_count,
            effective_batch_size=self._positions_this_step * gpu_count,  # positions_this_step is usually local batch
        )
        
        # Write to files
        self._write_jsonl(entry)
        self._write_csv(entry)
        self._write_tensorboard(entry)
        
        # Write to W&B
        if self.log_to_wandb:
            wandb.log({
                # Learning
                "learning/loss_total": entry.total_loss,
                "learning/loss_policy": entry.policy_loss,
                "learning/loss_value": entry.value_loss,
                "learning/entropy": entry.entropy,
                "learning/lr": entry.lr,
                "learning/grad_norm": entry.grad_norm,
                
                # Speed
                "speed/positions_per_sec": entry.positions_per_sec,
                "speed/step_time_ms": entry.step_time_ms,
                
                # Compute
                "compute/mfu": entry.mfu_percent,
                "compute/tflops": entry.tflops_per_sec,
                "compute/gpu_util": entry.gpu_util_percent,
                "compute/vram_mb": entry.gpu_mem_mb,
                
                # Scaling
                "scaling/gpu_count": entry.gpu_count,
                "scaling/effective_batch_size": entry.effective_batch_size,
                
                # System
                "system/cpu_util": entry.cpu_util_percent,
                "system/cpu_mem": entry.cpu_mem_percent,
            }, step=step)
        
        # Print to stdout
        if self.log_to_stdout and step % self.print_every == 0:
            self._print_train_step(entry)
    
    def _print_train_step(self, entry: TrainLogEntry):
        """Pretty print training step to stdout."""
        overflow_str = " [OVERFLOW]" if entry.amp_overflow else ""
        
        print(
            f"Step {entry.step:6d} | "
            f"Loss: {entry.total_loss:.4f} (P:{entry.policy_loss:.4f}) | "
            f"LR: {entry.lr:.2e} | "
            f"MFU: {entry.mfu_percent:.1f}% ({entry.tflops_per_sec:.1f} TF) | "
            f"Speed: {entry.positions_per_sec:.0f} pos/s | "
            f"GPU: {entry.gpu_mem_mb:.0f}MB{overflow_str}"
        )
    
    def log_eval(
        self,
        step: int,
        eval_type: str,
        policy_accuracy: float = 0.0,
        value_mse: float = 0.0,
        num_samples: int = 0,
        elo_estimate: float = 0.0,
        games_played: int = 0,
        wins: int = 0,
        draws: int = 0,
        losses: int = 0,
        engine_name: str = "",
        engine_settings: str = "",
    ):
        """Log evaluation results."""
        score = (wins + 0.5 * draws) / games_played if games_played > 0 else 0.0
        
        entry = EvalLogEntry(
            timestamp=datetime.now().isoformat(),
            step=step,
            eval_type=eval_type,
            policy_accuracy=policy_accuracy,
            value_mse=value_mse,
            num_samples=num_samples,
            elo_estimate=elo_estimate,
            games_played=games_played,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            engine_name=engine_name,
            engine_settings=engine_settings,
        )
        
        self._write_jsonl(entry)
        
        if self._tb_writer:
            if eval_type == "offline":
                self._tb_writer.add_scalar('eval/policy_accuracy', policy_accuracy, step)
                self._tb_writer.add_scalar('eval/value_mse', value_mse, step)
            else:
                self._tb_writer.add_scalar('eval/elo', elo_estimate, step)
                self._tb_writer.add_scalar('eval/score', score, step)
        
        if self.log_to_wandb:
            if eval_type == "offline":
                wandb.log({
                    "learning/eval_policy_accuracy": policy_accuracy,
                    "learning/eval_value_mse": value_mse,
                }, step=step)
            else:
                wandb.log({
                    "learning/elo": elo_estimate,
                    "learning/win_rate": score,
                    "learning/elo_games": games_played,
                }, step=step)
        
        if self.log_to_stdout:
            self._print_eval(entry)
    
    def _print_eval(self, entry: EvalLogEntry):
        """Pretty print evaluation results."""
        if entry.eval_type == "offline":
            print(f"\n{'='*60}")
            print(f"Offline Eval @ Step {entry.step}")
            print(f"  Policy Accuracy: {entry.policy_accuracy:.2%}")
            print(f"  Value MSE: {entry.value_mse:.4f}")
            print(f"  Samples: {entry.num_samples:,}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"Elo Eval @ Step {entry.step}")
            print(f"  vs {entry.engine_name} ({entry.engine_settings})")
            print(f"  Games: {entry.games_played} | W:{entry.wins} D:{entry.draws} L:{entry.losses}")
            print(f"  Score: {entry.score:.1%}")
            print(f"  Estimated Elo: {entry.elo_estimate:.0f}")
            print(f"{'='*60}\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "step_time_mean_ms": self.step_times.mean,
            "step_time_p95_ms": self.step_times.percentile(95),
            "throughput_ema": self.throughput_ema.value or 0.0,
        }
    
    def close(self):
        """Close all file handles."""
        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        if self._tb_writer:
            self._tb_writer.close()
            self._tb_writer = None
        if self.log_to_wandb:
            wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# ============================================================================
# Progress bar wrapper
# ============================================================================

def create_progress_bar(iterable, total=None, desc=None, disable=False):
    """Create a progress bar, falls back gracefully if tqdm unavailable."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, disable=disable)
    except ImportError:
        return iterable


# ============================================================================
# Testing
# ============================================================================

def test_logger():
    """Test logger functionality."""
    import tempfile
    import shutil
    
    print("Testing MetricsLogger...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(
            log_dir=tmpdir,
            experiment_name="test",
            log_to_tensorboard=False,
        )
        
        # Simulate training steps
        for step in range(10):
            logger.start_step(batch_size=256)
            logger.timers.start('data')
            time.sleep(0.01)
            logger.timers.stop('data')
            logger.timers.start('compute')
            time.sleep(0.02)
            logger.timers.stop('compute')
            
            logger.log_train_step(
                step=step,
                policy_loss=1.0 - step * 0.05,
                value_loss=0.5 - step * 0.02,
                total_loss=1.5 - step * 0.07,
                lr=0.001,
                grad_norm=1.0,
            )
        
        # Log eval
        logger.log_eval(
            step=10,
            eval_type="offline",
            policy_accuracy=0.35,
            value_mse=0.25,
            num_samples=10000,
        )
        
        logger.log_eval(
            step=10,
            eval_type="elo",
            elo_estimate=1200,
            games_played=100,
            wins=30,
            draws=40,
            losses=30,
            engine_name="Stockfish",
            engine_settings="depth=10",
        )
        
        logger.close()
        
        # Verify files created
        assert os.path.exists(os.path.join(tmpdir, "test.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "test.csv"))
        
        # Read and verify JSONL
        with open(os.path.join(tmpdir, "test.jsonl")) as f:
            lines = f.readlines()
            assert len(lines) == 12  # 10 train + 2 eval
        
        print("✓ Logger test passed!")


def test_timers():
    """Test timer utilities."""
    print("Testing timers...")
    
    # Simple timer
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    elapsed = timer.stop()
    assert 0.09 < elapsed < 0.15, f"Timer mismatch: {elapsed}"
    
    # Timer collection
    timers = TimerCollection()
    timers.start('a')
    time.sleep(0.05)
    timers.stop('a')
    timers.start('b')
    time.sleep(0.05)
    timers.stop('b')
    
    all_times = timers.get_all()
    assert 'a' in all_times and 'b' in all_times
    
    print("✓ Timer tests passed!")


if __name__ == "__main__":
    test_timers()
    print()
    test_logger()
