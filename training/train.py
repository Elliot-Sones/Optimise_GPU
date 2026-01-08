#!/usr/bin/env python3
"""
Training script for AlphaZero-style chess network.

Features:
- Mixed precision training (AMP)
- Cosine LR schedule with warmup
- Gradient clipping
- Checkpointing with full resume
- DDP multi-GPU support (via torchrun)
- Comprehensive logging
"""

import os
import sys
import argparse
import time
import random
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import (
    ChessIterableDataset, 
    GameFilter, 
    create_dataloader,
)
from models import ChessResNet, create_model, count_parameters
from moves import NUM_ACTIONS
from utils_logging import MetricsLogger, create_progress_bar


# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AlphaZero-style chess network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument("--pgn-files", type=str, nargs="+", required=True,
                        help="Paths to .pgn.zst training files")
    parser.add_argument("--eval-pgn", type=str, default=None,
                        help="Path to .pgn.zst eval file")
    parser.add_argument("--min-plies", type=int, default=10,
                        help="Minimum game length in plies")
    parser.add_argument("--min-rating", type=int, default=None,
                        help="Minimum average rating filter")
    parser.add_argument("--subsample-k", type=int, default=1,
                        help="Use every K-th position")
    
    # Model
    parser.add_argument("--model-variant", type=str, default="medium",
                        choices=["tiny", "small", "medium", "large"],
                        help="Model size variant")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="Override number of residual blocks")
    parser.add_argument("--channels", type=int, default=None,
                        help="Override number of channels")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per GPU")
    parser.add_argument("--total-steps", type=int, default=100000,
                        help="Total training steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--policy-loss-weight", type=float, default=1.0,
                        help="Weight for policy loss")
    parser.add_argument("--value-loss-weight", type=float, default=1.0,
                        help="Weight for value loss")
    
    # AMP
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    
    # torch.compile
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Save checkpoint every N steps")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="chess-alpha-zero",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (username/team)")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B run group")
    
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="Offline eval every N steps (0 to disable)")
    parser.add_argument("--elo-every", type=int, default=0,
                        help="Elo eval every N steps (0 to disable)")
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish binary for Elo eval")
    parser.add_argument("--elo-games", type=int, default=100,
                        help="Number of games for Elo eval")
    
    # DataLoader
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="DataLoader prefetch factor")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--overfit-test", action="store_true",
                        help="Overfit on small data (for testing)")
    
    return parser.parse_args()


# ============================================================================
# Distributed setup
# ============================================================================

def setup_distributed():
    """Setup distributed training if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    step: int,
    epoch: int,
    args: argparse.Namespace,
    best_loss: float = float('inf'),
):
    """Save training checkpoint."""
    # Handle DDP
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'args': vars(args),
        'best_loss': best_loss,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    
    torch.save(checkpoint, path)
    
    # Also save a "latest" symlink/copy
    latest_path = os.path.join(os.path.dirname(path), "latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    # Load model
    model_state = checkpoint['model_state_dict']
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    # Load optimizer
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler
    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Restore RNG state
    if 'rng_state' in checkpoint:
        rng = checkpoint['rng_state']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    
    return checkpoint


# ============================================================================
# Learning rate schedule
# ============================================================================

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create cosine LR schedule with linear warmup."""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-7,
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


# ============================================================================
# Training step
# ============================================================================

def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    grad_clip: float,
    policy_weight: float,
    value_weight: float,
    device: torch.device,
    use_amp: bool,
    use_amp: bool,
    logger: MetricsLogger,
) -> Tuple[float, float, float, float, float]:
    """
    Execute single training step.
    
    Returns:
        policy_loss, value_loss, total_loss, grad_norm, entropy
    """
    boards, policy_targets, value_targets = batch[:3]
    
    boards = boards.to(device, non_blocking=True)
    policy_targets = policy_targets.to(device, non_blocking=True)
    value_targets = value_targets.to(device, non_blocking=True)
    
    # Get legal mask if provided
    legal_mask = None
    if len(batch) > 3:
        legal_mask = batch[3].to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    logger.timers.start('compute')
    
    with autocast(device_type='cuda', enabled=use_amp):
        policy_logits, value_pred = model(boards, legal_mask)
        
        # Policy loss: cross-entropy with one-hot targets
        # Convert one-hot to class indices
        policy_targets_idx = policy_targets.argmax(dim=-1)
        policy_loss = F.cross_entropy(policy_logits, policy_targets_idx)
        
        # Entropy (for monitoring exploration/collapse)
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Value loss: MSE
        value_loss = F.mse_loss(value_pred, value_targets)
        
        # Combined loss
        total_loss = policy_weight * policy_loss + value_weight * value_loss
    
    logger.timers.stop('compute')
    logger.timers.start('backward')
    
    # Backward pass
    if scaler:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    
    logger.timers.stop('backward')
    
    return (
        policy_loss.item(),
        value_loss.item(),
        total_loss.item(),
        grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        entropy.item(),
    )


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_offline(
    model: nn.Module,
    eval_loader,
    device: torch.device,
    max_batches: int = 100,
) -> Tuple[float, float, int]:
    """
    Offline evaluation: policy accuracy and value MSE.
    
    Returns:
        policy_accuracy, value_mse, num_samples
    """
    model.eval()
    
    total_correct = 0
    total_mse = 0.0
    total_samples = 0
    
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        
        boards, policy_targets, value_targets = batch[:3]
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        
        legal_mask = None
        if len(batch) > 3:
            legal_mask = batch[3].to(device)
        
        policy_logits, value_pred = model(boards, legal_mask)
        
        # Policy accuracy
        policy_preds = policy_logits.argmax(dim=-1)
        policy_targets_idx = policy_targets.argmax(dim=-1)
        total_correct += (policy_preds == policy_targets_idx).sum().item()
        
        # Value MSE
        total_mse += F.mse_loss(value_pred, value_targets, reduction='sum').item()
        
        total_samples += boards.size(0)
    
    model.train()
    
    if total_samples == 0:
        return 0.0, 0.0, 0
    
    return (
        total_correct / total_samples,
        total_mse / total_samples,
        total_samples,
    )


# ============================================================================
# Main training loop
# ============================================================================

def train(args):
    """Main training function."""
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    
    # Create model
    model_kwargs = {}
    if args.num_blocks:
        model_kwargs['num_blocks'] = args.num_blocks
    if args.channels:
        model_kwargs['channels'] = args.channels
    
    model = create_model(args.model_variant, **model_kwargs)
    model = model.to(device)
    
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"AlphaZero Chess Training")
        print(f"{'='*60}")
        print(f"Model: {args.model_variant} ({count_parameters(model):,} params)")
        
        # Calculate FLOPs
        from models import estimate_flops
        flops_per_pos = estimate_flops(model)
        peak_flops = get_gpu_peak_flops()
        print(f"FLOPs per position: {flops_per_pos:.2e}")
        print(f"Approximated GPU Peak TFLOPS: {peak_flops/1e12:.1f}")
        
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        print(f"{'='*60}\n")
    else:
        # Worker processes need these too
        from models import estimate_flops
        flops_per_pos = estimate_flops(model)
        peak_flops = get_gpu_peak_flops()


def get_gpu_peak_flops() -> float:
    """Estimated Peak FLOPS for common GPUs (FP16 Tensor Core)."""
    if not torch.cuda.is_available():
        return 1.0
    
    name = torch.cuda.get_device_name(0).lower()
    
    # FP16 Tensor Core Peak estimations (approximate)
    if "4090" in name: return 330e12  # ~83 TFLOPS * ~4 (Tensor Core / Sparsity) -> simpler: 165 dense
    if "a100" in name: return 312e12
    if "3090" in name: return 142e12
    if " t4" in name: return 65e12
    if "v100" in name: return 125e12
    
    # Default fallback: 100 TFLOPS
    return 100e12
    
    # torch.compile
    if args.compile and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create data loaders
    game_filter = GameFilter(
        min_plies=args.min_plies,
        min_rating=args.min_rating,
    )
    
    train_loader = create_dataloader(
        pgn_files=args.pgn_files,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        game_filter=game_filter,
        subsample_every_k=args.subsample_k,
        include_legal_mask=True,
        prefetch_factor=args.prefetch_factor,
    )
    
    eval_loader = None
    if args.eval_pgn:
        eval_loader = create_dataloader(
            pgn_files=[args.eval_pgn],
            batch_size=args.batch_size,
            num_workers=2,
            include_legal_mask=True,
        )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    scheduler = create_lr_scheduler(optimizer, args.warmup_steps, args.total_steps)
    
    # AMP scaler
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Logger
    logger = None
    if is_main_process():
        logger = MetricsLogger(
            log_dir=args.log_dir,
            experiment_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
            log_to_tensorboard=args.tensorboard,
            log_to_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
            config=vars(args),
            print_every=args.log_every,
        )
    
    # Resume from checkpoint
    start_step = 0
    epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if is_main_process():
            print(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scaler, device)
        start_step = checkpoint['step'] + 1
        epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Advance scheduler to correct step
        for _ in range(start_step):
            scheduler.step()
    
    # Create checkpoint directory
    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    model.train()
    step = start_step
    data_iter = iter(train_loader)
    
    if is_main_process():
        print(f"Starting training from step {start_step}...")
    
    while step < args.total_steps:
        # Get batch
        if logger:
            logger.start_step(
                batch_size=args.batch_size,
                flops_per_position=flops_per_pos,
                gpu_peak_flops=peak_flops
            )
            logger.timers.start('data')
        
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        if logger:
            logger.timers.stop('data')
        
        # Training step
        policy_loss, value_loss, total_loss, grad_norm, entropy = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            policy_weight=args.policy_loss_weight,
            value_weight=args.value_loss_weight,
            device=device,
            use_amp=use_amp,
            logger=logger if logger else MetricsLogger.__new__(MetricsLogger),
        )
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        if logger and step % args.log_every == 0:
            amp_scale = scaler.get_scale() if scaler else 1.0
            amp_overflow = scaler._found_inf_per_device(optimizer) if scaler else False
            
            logger.log_train_step(
                step=step,
                epoch=epoch,
                policy_loss=policy_loss,
                value_loss=value_loss,
                total_loss=total_loss,
                lr=current_lr,
                lr=current_lr,
                grad_norm=grad_norm,
                amp_scale=amp_scale if isinstance(amp_scale, float) else amp_scale.item(),
                amp_overflow=bool(amp_overflow) if not isinstance(amp_overflow, bool) else amp_overflow,
                entropy=entropy,
                gpu_count=world_size,
            )
        
        # Checkpointing
        if is_main_process() and step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, epoch, args, best_loss)
            print(f"Saved checkpoint: {ckpt_path}")
        
        # Offline evaluation
        if eval_loader and args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
            if is_main_process():
                accuracy, mse, num_samples = evaluate_offline(model, eval_loader, device)
                if logger:
                    logger.log_eval(
                        step=step,
                        eval_type="offline",
                        policy_accuracy=accuracy,
                        value_mse=mse,
                        num_samples=num_samples,
                    )
        
        # Elo evaluation
        if args.elo_every > 0 and step > 0 and step % args.elo_every == 0:
            if is_main_process() and args.stockfish_path:
                try:
                    from eval import run_elo_eval
                    result = run_elo_eval(
                        model=model.module if hasattr(model, 'module') else model,
                        stockfish_path=args.stockfish_path,
                        num_games=args.elo_games,
                        device=device,
                    )
                    if logger:
                        logger.log_eval(
                            step=step,
                            eval_type="elo",
                            elo_estimate=result['elo'],
                            games_played=result['games'],
                            wins=result['wins'],
                            draws=result['draws'],
                            losses=result['losses'],
                            engine_name="Stockfish",
                            engine_settings=result.get('settings', ''),
                        )
                except Exception as e:
                    print(f"Elo eval failed: {e}")
        
        step += 1
        
        # Overfit test: stop early if loss is low enough
        if args.overfit_test and total_loss < 0.1:
            if is_main_process():
                print(f"Overfit test passed! Loss: {total_loss:.4f}")
            break
    
    # Final checkpoint
    if is_main_process():
        ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scaler, step, epoch, args, best_loss)
        print(f"Training complete! Final checkpoint: {ckpt_path}")
        
        if logger:
            logger.close()
    
    cleanup_distributed()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
