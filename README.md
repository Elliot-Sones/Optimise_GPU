# AlphaZero Chess Training

Fast, reproducible PyTorch codebase for training an AlphaZero-style chess network from Lichess PGN data.

## Features

- **Streaming dataset**: Reads `.pgn.zst` via streaming decompression (never stores full dataset)
- **AlphaZero architecture**: 20-block ResNet with policy (4672 actions) and value heads
- **Mixed precision training**: AMP with gradient scaling
- **Multi-GPU support**: DDP via `torchrun`
- **Comprehensive logging**: stdout, CSV, JSONL, TensorBoard
- **Elo evaluation**: Play against Stockfish with fixed openings

## Installation

```bash
cd training
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, CUDA-capable GPU recommended

### 1. Download Data
Download Lichess game database (approx 30GB compressed):
```bash
mkdir -p data
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst -P data/
```

### 2. Prepare Validation Split (Optional but Recommended)
Extract 5,000 games to track how well your model generalizes:
```bash
python3 scripts/make_val_split.py data/lichess_db_standard_rated_2024-01.pgn.zst data/val.pgn.zst 5000
```

### 3. Start Training (Optimized & Monitored)
Track everything with **Weights & Biases** (requires `pip install wandb`):

```bash
# Login once
wandb login

# Single GPU Training
python3 training/train.py \
    --pgn-files data/lichess_db_standard_rated_2024-01.pgn.zst \
    --eval-pgn data/val.pgn.zst \
    --eval-every 5000 \
    --batch-size 1024 \
    --compile \
    --lr 2e-4 \
    --total-steps 100000 \
    --wandb \
    --wandb-project chess-training
```

**Key Flags Explained:**
*   `--compile`: Uses PyTorch 2.0+ optimization (huge speedup).
*   `--wandb`: Live dashboards for Loss, **MFU** (Efficiency), and Hardware Health.
*   `--eval-pgn`: Checks accuracy on unseen games every 5k steps.

### Monitoring
Check your W&B dashboard for:
*   **Throughput/MFU**: Aim for >30%.
*   **System/GPU Util**: Should stay near 100%.
*   **Eval/Policy Accuracy**: Should climb steadily.



### Training (Multi-GPU)

```bash
cd training
torchrun --nproc_per_node=4 train.py --pgn-files ../data/*.pgn.zst --batch-size 256
```

### Resume Training

```bash
python3 training/train.py --pgn-files data/*.pgn.zst --resume checkpoints/latest.pt
```

### Offline Evaluation

```bash
python3 training/eval.py --mode offline --model checkpoints/step_50000.pt --eval-pgn data/test.pgn.zst
```

### Elo Evaluation

```bash
python3 training/eval.py --mode elo --model checkpoints/step_50000.pt \
    --stockfish-path /usr/local/bin/stockfish --num-games 100
```

### Analyze Training Logs

```bash
python3 training/analyze_logs.py logs/20240101_120000.jsonl --plot
```

## Data Preparation

Download Lichess games from [database.lichess.org](https://database.lichess.org/#standard_games):

```bash
mkdir -p data
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst -P data/
```

## Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--pgn-files` | required | Paths to `.pgn.zst` training files |
| `--batch-size` | 256 | Batch size per GPU |
| `--lr` | 2e-4 | Peak learning rate |
| `--total-steps` | 100000 | Total training steps |
| `--warmup-steps` | 1000 | LR warmup steps |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--model-variant` | medium | Model size (tiny/small/medium/large) |
| `--min-rating` | None | Filter games by minimum rating |
| `--compile` | False | Use torch.compile (PyTorch 2.0+) |

## Project Structure

```
├── training/
│   ├── train.py          # Main training script
│   ├── data.py           # Streaming PGN dataset
│   ├── models.py         # ResNet architecture
│   ├── moves.py          # Move encoding/decoding
│   ├── eval.py           # Offline and Elo evaluation
│   ├── utils_logging.py  # Comprehensive logging
│   ├── analyze_logs.py   # Log analysis and plots
│   └── requirements.txt  # Dependencies
├── data/                 # PGN files (create this)
├── checkpoints/          # Model checkpoints (auto-created)
├── logs/                 # Training logs (auto-created)
└── README.md
```

## Sanity Tests

```bash
cd training

# Test move encoding round-trip
python3 moves.py

# Test model forward/backward
python3 models.py
```

## License

MIT