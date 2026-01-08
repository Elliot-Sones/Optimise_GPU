# AlphaZero Chess Training - Complete Technical Guide

This document explains exactly how the training system works, from raw PGN data to a trained chess neural network.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Data Pipeline](#data-pipeline)
3. [Move Encoding](#move-encoding)
4. [Board Representation](#board-representation)
5. [Neural Network Architecture](#neural-network-architecture)
6. [Training Process](#training-process)
7. [Evaluation](#evaluation)
8. [Complete Training Flow](#complete-training-flow)

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   .pgn.zst files          Neural Network              Trained Model │
│        │                       │                           │        │
│        ▼                       ▼                           ▼        │
│   ┌─────────┐            ┌──────────┐              ┌────────────┐   │
│   │ Stream  │──boards──▶ │  ResNet  │──loss──▶     │ Checkpoint │   │
│   │ Decomp. │            │  (GPU)   │              │   .pt      │   │
│   │ + Parse │──moves───▶ │          │              └────────────┘   │
│   └─────────┘            └──────────┘                               │
│        │                       │                                     │
│        ▼                       ▼                                     │
│   Never stored            Learns to                                  │
│   in memory               predict moves                              │
│                           + game outcome                             │
└─────────────────────────────────────────────────────────────────────┘
```

**The goal**: Train a neural network to look at a chess position and:
1. **Policy**: Predict what move a strong player would make
2. **Value**: Predict who will win from this position

---

## Data Pipeline

### Source: Lichess Games

We use games from [database.lichess.org](https://database.lichess.org). These are compressed `.pgn.zst` files containing millions of real games.

Example file: `lichess_db_standard_rated_2024-01.pgn.zst` (~30GB compressed, ~200GB uncompressed)

### Streaming Decompression (Never Load Full Dataset)

```python
# data.py - ZstdPGNReader

import zstandard as zstd
import chess.pgn

dctx = zstd.ZstdDecompressor()
with open("games.pgn.zst", 'rb') as fh:
    with dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        while True:
            game = chess.pgn.read_game(text_stream)  # One game at a time
            if game is None:
                break
            yield game  # Process immediately, don't store
```

**Key insight**: We decompress and parse one game at a time. A 30GB file never needs more than a few MB of RAM.

### From Games to Training Samples

Each game produces multiple training samples:

```
Game: 1. e4 e5 2. Nf3 Nc6 3. Bb5 ... (White wins)

Position 1: Starting position
  → Board tensor (18×8×8)
  → Move played: e4 (index 772)
  → Outcome: +1 (white won)

Position 2: After 1. e4
  → Board tensor (18×8×8)  
  → Move played: e5 (index 892)
  → Outcome: -1 (from black's view, white won)

Position 3: After 1. e4 e5
  → Board tensor (18×8×8)
  → Move played: Nf3 (index 1456)
  → Outcome: +1 (white won)

... and so on for every position in the game
```

### Filtering Options

The dataset can filter games:

```python
GameFilter(
    min_plies=10,       # Games must have at least 10 moves
    min_rating=1800,    # Both players rated 1800+
    time_control_pattern=r"600\+",  # Only 10 minute games
)
```

---

## Move Encoding

### The 4672-Action Space

Chess has variable numbers of legal moves per position (roughly 20-40). But neural networks need fixed-size outputs. AlphaZero uses **4672 actions**:

```
8×8×73 = 4672 possible moves
  │  │  │
  │  │  └── 73 move types from that square
  │  └───── 8 ranks
  └──────── 8 files
```

### The 73 Move Types

From any square, there are at most 73 ways to move:

```
Planes 0-55:  Queen-style moves (56 total)
              8 directions × 7 distances = 56
              
              N  NE  E  SE  S  SW  W  NW
              ↑  ↗   →  ↘   ↓  ↙   ←  ↖
              
              Each direction: 1-7 squares distance

Planes 56-63: Knight moves (8 total)
              All 8 possible knight jumps
              
Planes 64-72: Underpromotions (9 total)
              When pawn reaches 8th rank:
              3 pieces (knight, bishop, rook)
              × 3 directions (left-capture, straight, right-capture)
              
              Queen promotion uses planes 0-55 (it's a "queen-style" move)
```

### Example: Encoding e2-e4

```python
from moves import move_to_index
import chess

board = chess.Board()
move = chess.Move.from_uci("e2e4")

idx = move_to_index(move, board.turn)  # Returns 772

# Breakdown:
# From square e2 = file 4, rank 1 = square 12
# Direction: North (plane offset 0)
# Distance: 2 squares (plane = 0*7 + (2-1) = 1)
# Index = 12 * 73 + 1 = 877 (different if from white's view)
```

### Black's Perspective Flip

For consistency, we always encode from the current player's view:
- White: e2-e4 is "forward 2 squares"
- Black: e7-e5 is ALSO "forward 2 squares" (board flipped mentally)

This means the network learns a color-agnostic policy.

---

## Board Representation

### 18-Plane Encoding

The board is represented as 18 channels of 8×8 grids:

```
Planes 0-5:   Our pieces (current player)
              [0] Pawns    [1] Knights   [2] Bishops
              [3] Rooks    [4] Queens    [5] King

Planes 6-11:  Opponent pieces
              Same order as above

Planes 12-15: Castling rights (binary, all 64 squares same value)
              [12] We can castle kingside
              [13] We can castle queenside  
              [14] They can castle kingside
              [15] They can castle queenside

Plane 16:     En passant (single 1 on the EP square if available)

Plane 17:     Side to move (all 1s - we always see it as "our turn")
```

### Visual Example

Starting position, White to move:

```
Plane 0 (our pawns):        Plane 6 (their pawns):
. . . . . . . .             1 1 1 1 1 1 1 1
. . . . . . . .             . . . . . . . .
. . . . . . . .             . . . . . . . .
. . . . . . . .             . . . . . . . .
. . . . . . . .             . . . . . . . .
. . . . . . . .             . . . . . . . .
1 1 1 1 1 1 1 1             . . . . . . . .
. . . . . . . .             . . . . . . . .
```

---

## Neural Network Architecture

### Overview

```
Input: 18×8×8 tensor
         │
         ▼
    ┌─────────┐
    │  Stem   │  Conv2d 3×3, 18→256 channels, BatchNorm, ReLU
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Block 1 │ ─┐
    └────┬────┘  │
         │       │  20 residual blocks
         ▼       │  Each: Conv→BN→ReLU→Conv→BN + skip
    ┌─────────┐  │
    │   ...   │  │
    └────┬────┘  │
         │       │
    ┌─────────┐  │
    │Block 20 │ ─┘
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌──────┐
│Policy│  │Value │
│ Head │  │ Head │
└──┬───┘  └──┬───┘
   │         │
   ▼         ▼
4672 logits  Scalar [-1, 1]
```

### Residual Block Detail

```python
class ResidualBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))  # Conv→BN→ReLU
        out = self.bn2(self.conv2(out))            # Conv→BN
        out = self.relu(out + residual)            # Add skip, then ReLU
        return out
```

The skip connection lets gradients flow easily through deep networks.

### Policy Head

```python
# 256 channels → 32 channels (1×1 conv) → flatten → 4672 logits
Conv2d(256, 32, 1) → BatchNorm → ReLU → Flatten → Linear(32*64, 4672)
```

Output: 4672 raw logits. During training, we mask illegal moves to -infinity.

### Value Head

```python
# 256 channels → 1 channel (1×1 conv) → flatten → 256 hidden → 1 output
Conv2d(256, 1, 1) → BatchNorm → ReLU → Flatten → Linear(64, 256) → ReLU → Linear(256, 1) → Tanh
```

Output: Single value in [-1, 1] representing expected outcome.

### Model Sizes

| Variant | Blocks | Channels | Parameters |
|---------|--------|----------|------------|
| tiny    | 5      | 64       | ~200K      |
| small   | 10     | 128      | ~2M        |
| medium  | 20     | 256      | ~15M       |
| large   | 40     | 256      | ~30M       |

---

## Training Process

### Loss Function

We optimize two objectives simultaneously:

```python
# Policy loss: Cross-entropy
# "Did the network predict the move that was actually played?"
policy_loss = CrossEntropy(predicted_logits, actual_move_index)

# Value loss: Mean squared error  
# "Did the network predict the correct game outcome?"
value_loss = MSE(predicted_value, actual_outcome)

# Combined
total_loss = policy_loss + value_loss
```

### Training Loop (Simplified)

```python
for step in range(total_steps):
    # 1. Get batch of positions from streaming dataset
    boards, moves, outcomes = next(dataloader)  # Shape: (256, 18, 8, 8), (256,), (256,)
    
    # 2. Forward pass with mixed precision
    with autocast():
        policy_logits, values = model(boards)
        policy_loss = cross_entropy(policy_logits, moves)
        value_loss = mse_loss(values, outcomes)
        total_loss = policy_loss + value_loss
    
    # 3. Backward pass with gradient scaling
    scaler.scale(total_loss).backward()
    
    # 4. Gradient clipping (stability)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 5. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # 6. Learning rate schedule
    scheduler.step()
```

### Key Training Components

#### Mixed Precision (AMP)

Uses FP16 for most operations, FP32 for critical ones:
- **2x faster** forward/backward pass
- **~30% less memory** usage
- Automatic with `torch.cuda.amp`

```python
scaler = GradScaler()
with autocast():
    loss = model(inputs)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Learning Rate Schedule

Cosine decay with warmup:

```
LR
 │    ╱──────╲
 │   ╱        ╲
 │  ╱          ╲
 │ ╱            ╲
 │╱              ╲
 └───────────────────► Steps
   ↑            ↑
   warmup       decay
```

- **Warmup**: Gradually increase LR from ~0 to peak (prevents early instability)
- **Cosine decay**: Smoothly decrease LR to near-zero (fine-tune at end)

#### Gradient Clipping

Prevents exploding gradients by capping the total gradient norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Checkpointing

Saved every N steps:

```python
checkpoint = {
    'step': current_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'rng_state': {...}  # For exact reproducibility
}
torch.save(checkpoint, 'checkpoints/step_50000.pt')
```

To resume: `--resume checkpoints/step_50000.pt`

---

## Evaluation

### Offline Evaluation

Measures how well the network predicts human moves:

```python
# Policy accuracy: % of positions where network's top choice = actual move
accuracy = (argmax(policy_logits) == actual_move).mean()

# Value MSE: How close is predicted outcome to actual?
mse = mean((predicted_value - actual_outcome)^2)
```

**Typical values**:
- Random policy: ~0.3% accuracy (1/~350 legal moves)
- Trained model: 40-55% top-1 accuracy
- Value MSE: 0.2-0.4 (lower is better)

### Elo Evaluation

Plays actual games against Stockfish:

```
1. Load model checkpoint
2. For each game:
   a. Apply opening from fixed set (reproducibility)
   b. Alternate moves: NN vs Stockfish
   c. Continue until game over
3. Calculate score: wins + 0.5*draws
4. Estimate Elo from score vs Stockfish's known strength
```

**Elo formula**:
```
Elo_diff = 400 * log10(score / (1 - score))
Elo = Stockfish_Elo + Elo_diff
```

---

## Complete Training Flow

### Step-by-Step: What Happens When You Run Training

```bash
python3 train.py --pgn-files ../data/*.pgn.zst --batch-size 512 --total-steps 100000
```

**1. Initialization (first few seconds)**
```
- Parse command line arguments
- Set random seeds (reproducibility)
- Create model (20-block ResNet, ~15M params)
- Move model to GPU
- Create optimizer (AdamW)
- Create learning rate scheduler
- Create AMP scaler
- Initialize logging
```

**2. Data Pipeline Setup**
```
- Create IterableDataset pointing to .pgn.zst files
- Create DataLoader with 4 workers
- Each worker streams from different part of file
- Pin memory enabled for faster GPU transfer
```

**3. Training Loop (per step)**
```
Step 0:
├── DataLoader workers decompress next batch in parallel
├── Batch arrives: 512 positions (18×8×8), 512 move indices, 512 outcomes
├── Transfer to GPU (async, pinned memory)
├── Forward pass (fp16 with autocast)
│   ├── Stem: 18→256 channels
│   ├── 20 residual blocks
│   ├── Policy head → 512×4672 logits
│   └── Value head → 512 scalars
├── Compute loss (policy cross-entropy + value MSE)
├── Backward pass (fp16, scaled gradients)
├── Unscale gradients
├── Clip gradients (max norm 1.0)
├── Optimizer step (AdamW)
├── Scaler update (adjusts scale factor)
├── LR scheduler step
└── Log metrics (if step % 100 == 0)
```

**4. Periodic Events**
```
Every 100 steps:   Log to stdout + JSONL + CSV
Every 5000 steps:  Save checkpoint
Every 5000 steps:  Run offline evaluation
Every N steps:     Run Elo evaluation (if enabled)
```

**5. Metrics You'll See**
```
Step   1000 | Loss: 4.2341 (P:3.8123 V:0.4218) | LR: 1.00e-04 | Grad: 0.82 | Speed: 18432 pos/s | GPU: 8234MB
Step   2000 | Loss: 3.9876 (P:3.5432 V:0.4444) | LR: 1.50e-04 | Grad: 0.91 | Speed: 19201 pos/s | GPU: 8234MB
...
```

**6. End of Training**
```
- Final checkpoint saved
- Logs closed
- Model ready for evaluation or deployment
```

### Expected Training Progression

| Step | Policy Loss | Value Loss | Policy Acc | Approx Elo |
|------|-------------|------------|------------|------------|
| 0    | ~8.0        | ~0.5       | ~0.3%      | Random     |
| 1K   | ~5.5        | ~0.45      | ~10%       | ~600       |
| 10K  | ~3.5        | ~0.35      | ~30%       | ~1200      |
| 50K  | ~2.8        | ~0.28      | ~42%       | ~1600      |
| 100K | ~2.5        | ~0.25      | ~48%       | ~1800      |

---

## Hardware Expectations (4090)

| Metric | Expected Value |
|--------|----------------|
| Batch size | 512-1024 |
| Throughput | 15,000-25,000 pos/sec |
| GPU Memory | 8-12 GB used (24 GB available) |
| 100K steps at 20K/s | ~1.5 hours |
| Power draw | ~350W |

### Bottleneck Analysis

The training logs include timing breakdown:
- **Data time > 40%**: Increase `--num-workers`, use SSD
- **Compute time ~80%**: GPU-bound (optimal!)
- **High overflow count**: Lower LR or increase grad clip

---

## Summary

1. **Data flows**: PGN.zst → stream decompress → parse → encode board/move → batch → GPU
2. **Model learns**: Board tensor → policy (what move?) + value (who wins?)
3. **Training**: Minimize cross-entropy (policy) + MSE (value) with AMP + gradient clipping
4. **Evaluation**: Offline accuracy + Elo vs Stockfish
5. **Result**: A network that can play chess by picking the highest-probability legal move
