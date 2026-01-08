#!/usr/bin/env python3
"""
Evaluation module for AlphaZero chess network.

Features:
- Offline evaluation (policy accuracy, value MSE)
- Elo evaluation against Stockfish
- Fixed opening book for reproducibility
- UCI player wrapper for neural network
"""

import os
import sys
import argparse
import random
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

from moves import move_to_index, index_to_move, get_legal_mask, NUM_ACTIONS
from data import board_to_tensor_fast, ChessIterableDataset, GameFilter
from models import ChessResNet, create_model


# ============================================================================
# Opening book
# ============================================================================

# Fixed set of openings for reproducible Elo evaluation
# Each opening is a sequence of moves in UCI notation
FIXED_OPENINGS = [
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    # Sicilian Defense
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    # Queen's Gambit
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    # Queen's Gambit Accepted
    ["d2d4", "d7d5", "c2c4", "d5c4"],
    # King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    # Scotch Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    # English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6"],
    # London System
    ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3"],
    # Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    # Catalan
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    # Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3"],
    # Modern Defense
    ["e2e4", "g7g6", "d2d4", "f8g7"],
]


def get_opening(idx: int) -> List[chess.Move]:
    """Get opening moves by index."""
    opening_uci = FIXED_OPENINGS[idx % len(FIXED_OPENINGS)]
    return [chess.Move.from_uci(m) for m in opening_uci]


# ============================================================================
# Neural Network Player
# ============================================================================

class NNPlayer:
    """
    Chess player using neural network.
    
    Selects move with highest policy logit among legal moves.
    """
    
    def __init__(
        self,
        model: ChessResNet,
        device: torch.device,
        temperature: float = 0.0,
        use_value_for_tiebreak: bool = False,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.use_value_for_tiebreak = use_value_for_tiebreak
        self.model.eval()
    
    @torch.no_grad()
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get best move for position."""
        # Encode board
        board_tensor = board_to_tensor_fast(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        # Get legal mask
        legal_mask = get_legal_mask(board)
        legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)
        
        # Forward pass
        policy_logits, value = self.model(board_tensor, legal_mask)
        
        if self.temperature > 0:
            # Sample from policy
            probs = F.softmax(policy_logits / self.temperature, dim=-1)
            action_idx = torch.multinomial(probs, 1).item()
        else:
            # Greedy
            action_idx = policy_logits.argmax(dim=-1).item()
        
        # Convert to move
        return index_to_move(action_idx, board)
    
    def get_value(self, board: chess.Board) -> float:
        """Get value prediction for position."""
        board_tensor = board_to_tensor_fast(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.model(board_tensor)
        
        return value.item()


# ============================================================================
# Elo Evaluation
# ============================================================================

@dataclass
class EloResult:
    """Result of Elo evaluation."""
    elo: float
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    settings: str
    pgn_games: List[str]


def calculate_elo(
    score: float,
    opponent_elo: float = 1500,
) -> float:
    """
    Calculate estimated Elo from score against known opponent.
    
    Uses simplified formula: Elo = opponent_elo + 400 * log10(score / (1 - score))
    """
    if score <= 0:
        return opponent_elo - 400
    if score >= 1:
        return opponent_elo + 400
    
    import math
    return opponent_elo + 400 * math.log10(score / (1 - score))


def play_game(
    nn_player: NNPlayer,
    engine: chess.engine.SimpleEngine,
    opening: List[chess.Move],
    nn_is_white: bool,
    time_limit: float = 0.1,
    depth_limit: int = 10,
) -> Tuple[str, str]:
    """
    Play a single game between NN and engine.
    
    Returns:
        (result, pgn_string) where result is from NN's perspective
    """
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    
    # Play opening
    for move in opening:
        if move in board.legal_moves:
            board.push(move)
            node = node.add_variation(move)
        else:
            break
    
    # Play game
    while not board.is_game_over():
        is_nn_turn = (board.turn == chess.WHITE) == nn_is_white
        
        if is_nn_turn:
            move = nn_player.get_move(board)
        else:
            result = engine.play(
                board,
                chess.engine.Limit(time=time_limit, depth=depth_limit),
            )
            move = result.move
        
        board.push(move)
        node = node.add_variation(move)
    
    # Determine result from NN's perspective
    outcome = board.outcome()
    if outcome is None:
        nn_result = "1/2-1/2"
    elif outcome.winner is None:
        nn_result = "1/2-1/2"
    elif (outcome.winner == chess.WHITE) == nn_is_white:
        nn_result = "1-0"
    else:
        nn_result = "0-1"
    
    # Set game headers
    game.headers["Event"] = "Elo Eval"
    game.headers["White"] = "NN" if nn_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if nn_is_white else "NN"
    game.headers["Result"] = board.result()
    
    return nn_result, str(game)


def run_elo_eval(
    model: ChessResNet,
    stockfish_path: str,
    num_games: int = 100,
    device: torch.device = None,
    time_limit: float = 0.1,
    depth_limit: int = 10,
    stockfish_elo: int = 1500,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Elo evaluation against Stockfish.
    
    Args:
        model: Chess network
        stockfish_path: Path to Stockfish binary
        num_games: Number of games to play
        device: Torch device
        time_limit: Engine time limit per move
        depth_limit: Engine depth limit
        stockfish_elo: Assumed Stockfish Elo at these settings
        seed: Random seed for opening selection
        verbose: Print progress
    
    Returns:
        Dictionary with Elo estimate and game statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    nn_player = NNPlayer(model, device)
    
    # Start engine
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Configure engine for consistent play
    try:
        engine.configure({
            "Threads": 1,
            "Hash": 64,
        })
    except:
        pass  # Some engines don't support these options
    
    random.seed(seed)
    
    wins = 0
    draws = 0
    losses = 0
    pgn_games = []
    
    if verbose:
        print(f"\nPlaying {num_games} games against Stockfish...")
        print(f"Engine settings: time={time_limit}s, depth={depth_limit}")
    
    for game_idx in range(num_games):
        # Alternate colors
        nn_is_white = (game_idx % 2 == 0)
        
        # Select opening
        opening_idx = game_idx // 2  # Same opening for both colors
        opening = get_opening(opening_idx)
        
        result, pgn = play_game(
            nn_player=nn_player,
            engine=engine,
            opening=opening,
            nn_is_white=nn_is_white,
            time_limit=time_limit,
            depth_limit=depth_limit,
        )
        
        pgn_games.append(pgn)
        
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1
        
        if verbose and (game_idx + 1) % 10 == 0:
            current_score = (wins + 0.5 * draws) / (game_idx + 1)
            print(f"  Game {game_idx + 1}/{num_games}: W={wins} D={draws} L={losses} Score={current_score:.1%}")
    
    engine.quit()
    
    # Calculate final score and Elo
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total > 0 else 0.5
    elo = calculate_elo(score, stockfish_elo)
    
    settings = f"time={time_limit}s,depth={depth_limit},elo={stockfish_elo}"
    
    if verbose:
        print(f"\nResults: W={wins} D={draws} L={losses}")
        print(f"Score: {score:.1%}")
        print(f"Estimated Elo: {elo:.0f}")
    
    return {
        'elo': elo,
        'games': total,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'score': score,
        'settings': settings,
        'pgn_games': pgn_games,
    }


# ============================================================================
# Offline Evaluation
# ============================================================================

@torch.no_grad()
def run_offline_eval(
    model: ChessResNet,
    eval_pgn: str,
    device: torch.device = None,
    max_samples: int = 10000,
    batch_size: int = 256,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run offline evaluation on held-out games.
    
    Args:
        model: Chess network
        eval_pgn: Path to evaluation .pgn.zst file
        device: Torch device
        max_samples: Maximum samples to evaluate
        batch_size: Batch size
        verbose: Print progress
    
    Returns:
        Dictionary with policy accuracy and value MSE
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    from data import create_dataloader
    
    eval_loader = create_dataloader(
        pgn_files=[eval_pgn],
        batch_size=batch_size,
        num_workers=2,
        include_legal_mask=True,
    )
    
    total_correct = 0
    total_mse = 0.0
    total_samples = 0
    
    if verbose:
        print(f"\nRunning offline evaluation on {eval_pgn}...")
    
    for batch in eval_loader:
        boards, policy_targets, value_targets, legal_masks = batch
        
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        legal_masks = legal_masks.to(device)
        
        policy_logits, value_pred = model(boards, legal_masks)
        
        # Policy accuracy
        policy_preds = policy_logits.argmax(dim=-1)
        policy_targets_idx = policy_targets.argmax(dim=-1)
        total_correct += (policy_preds == policy_targets_idx).sum().item()
        
        # Value MSE
        total_mse += F.mse_loss(value_pred, value_targets, reduction='sum').item()
        
        total_samples += boards.size(0)
        
        if total_samples >= max_samples:
            break
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    mse = total_mse / total_samples if total_samples > 0 else 0.0
    
    if verbose:
        print(f"Samples: {total_samples:,}")
        print(f"Policy Accuracy: {accuracy:.2%}")
        print(f"Value MSE: {mse:.4f}")
    
    return {
        'policy_accuracy': accuracy,
        'value_mse': mse,
        'num_samples': total_samples,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate chess network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["offline", "elo"],
                        help="Evaluation mode")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Offline eval
    parser.add_argument("--eval-pgn", type=str, default=None,
                        help="Path to evaluation .pgn.zst file")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Maximum samples for offline eval")
    
    # Elo eval
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish binary")
    parser.add_argument("--num-games", type=int, default=100,
                        help="Number of games for Elo eval")
    parser.add_argument("--time-limit", type=float, default=0.1,
                        help="Engine time limit per move")
    parser.add_argument("--depth-limit", type=int, default=10,
                        help="Engine depth limit")
    parser.add_argument("--stockfish-elo", type=int, default=1500,
                        help="Assumed Stockfish Elo at these settings")
    
    # Model
    parser.add_argument("--model-variant", type=str, default="medium",
                        help="Model variant if loading from scratch")
    
    # Output
    parser.add_argument("--save-pgn", type=str, default=None,
                        help="Save games to PGN file (Elo mode)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Get model config from checkpoint
    ckpt_args = checkpoint.get('args', {})
    model_variant = ckpt_args.get('model_variant', args.model_variant)
    num_blocks = ckpt_args.get('num_blocks')
    channels = ckpt_args.get('channels')
    
    model_kwargs = {}
    if num_blocks:
        model_kwargs['num_blocks'] = num_blocks
    if channels:
        model_kwargs['channels'] = channels
    
    model = create_model(model_variant, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if args.mode == "offline":
        if not args.eval_pgn:
            print("Error: --eval-pgn required for offline mode")
            sys.exit(1)
        
        result = run_offline_eval(
            model=model,
            eval_pgn=args.eval_pgn,
            device=device,
            max_samples=args.max_samples,
        )
        
        print(f"\n{'='*40}")
        print("Offline Evaluation Results")
        print(f"{'='*40}")
        print(f"Policy Accuracy: {result['policy_accuracy']:.2%}")
        print(f"Value MSE: {result['value_mse']:.4f}")
        print(f"Samples: {result['num_samples']:,}")
        
    elif args.mode == "elo":
        if not args.stockfish_path:
            print("Error: --stockfish-path required for elo mode")
            sys.exit(1)
        
        result = run_elo_eval(
            model=model,
            stockfish_path=args.stockfish_path,
            num_games=args.num_games,
            device=device,
            time_limit=args.time_limit,
            depth_limit=args.depth_limit,
            stockfish_elo=args.stockfish_elo,
            seed=args.seed,
        )
        
        print(f"\n{'='*40}")
        print("Elo Evaluation Results")
        print(f"{'='*40}")
        print(f"Games: {result['games']}")
        print(f"Score: {result['score']:.1%} (W={result['wins']} D={result['draws']} L={result['losses']})")
        print(f"Estimated Elo: {result['elo']:.0f}")
        
        # Save PGN if requested
        if args.save_pgn:
            with open(args.save_pgn, 'w') as f:
                for pgn in result['pgn_games']:
                    f.write(pgn)
                    f.write("\n\n")
            print(f"Games saved to: {args.save_pgn}")


if __name__ == "__main__":
    main()
