"""
Streaming dataset for chess positions from .pgn.zst files.

Features:
- Streaming decompression (never stores full dataset)
- Game filtering (min plies, rating, time control)
- Position subsampling
- Worker-safe for DataLoader
- Efficient board encoding
"""

import io
import re
import zstandard as zstd
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import Optional, Iterator, Tuple, List, Callable
from dataclasses import dataclass
import time

from moves import move_to_index, get_legal_mask, NUM_ACTIONS


# ============================================================================
# Board encoding
# ============================================================================

# 18 planes:
#  0-5: White pieces (pawn, knight, bishop, rook, queen, king)
#  6-11: Black pieces
#  12: White can castle kingside
#  13: White can castle queenside
#  14: Black can castle kingside
#  15: Black can castle queenside
#  16: En passant square (if any)
#  17: Side to move (all 1s if white, 0s if black)

NUM_PLANES_BOARD = 18
PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board as an 18x8x8 numpy array.
    
    The encoding is from the current player's perspective:
    - Board is flipped for black
    - Planes 0-5 are always "our" pieces, 6-11 "their" pieces
    """
    planes = np.zeros((NUM_PLANES_BOARD, 8, 8), dtype=np.float32)
    
    # Determine perspective
    our_color = board.turn
    their_color = not board.turn
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Get coordinates (flip board for black)
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if our_color == chess.BLACK:
            rank = 7 - rank
        
        # Determine plane offset
        if piece.color == our_color:
            plane_offset = 0
        else:
            plane_offset = 6
        
        plane = plane_offset + PIECE_PLANES[piece.piece_type]
        planes[plane, rank, file] = 1.0
    
    # Castling rights (from current player's perspective)
    if our_color == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15, :, :] = 1.0
    
    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        if our_color == chess.BLACK:
            rank = 7 - rank
        planes[16, rank, file] = 1.0
    
    # Side to move (always 1 from current player's perspective)
    planes[17, :, :] = 1.0
    
    return planes


def board_to_tensor_fast(board: chess.Board) -> np.ndarray:
    """
    Faster board encoding using piece maps.
    """
    planes = np.zeros((NUM_PLANES_BOARD, 8, 8), dtype=np.float32)
    our_color = board.turn
    flip = our_color == chess.BLACK
    
    # Encode pieces using piece_map for efficiency
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if flip:
            rank = 7 - rank
        
        plane_offset = 0 if piece.color == our_color else 6
        plane = plane_offset + PIECE_PLANES[piece.piece_type]
        planes[plane, rank, file] = 1.0
    
    # Castling (vectorized)
    our_k = board.has_kingside_castling_rights(our_color)
    our_q = board.has_queenside_castling_rights(our_color)
    their_k = board.has_kingside_castling_rights(not our_color)
    their_q = board.has_queenside_castling_rights(not our_color)
    
    if our_k:
        planes[12] = 1.0
    if our_q:
        planes[13] = 1.0
    if their_k:
        planes[14] = 1.0
    if their_q:
        planes[15] = 1.0
    
    # En passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        if flip:
            rank = 7 - rank
        planes[16, rank, file] = 1.0
    
    # Side to move
    planes[17] = 1.0
    
    return planes


# ============================================================================
# Game filtering
# ============================================================================

@dataclass
class GameFilter:
    """Configuration for filtering games."""
    min_plies: int = 10
    max_plies: Optional[int] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    time_control_pattern: Optional[str] = None  # Regex pattern
    require_both_ratings: bool = True
    
    def __post_init__(self):
        if self.time_control_pattern:
            self._tc_regex = re.compile(self.time_control_pattern)
        else:
            self._tc_regex = None
    
    def accept_game(self, game: chess.pgn.Game) -> bool:
        """Check if a game passes the filter."""
        headers = game.headers
        
        # Check ply count
        mainline = list(game.mainline_moves())
        num_plies = len(mainline)
        
        if num_plies < self.min_plies:
            return False
        if self.max_plies and num_plies > self.max_plies:
            return False
        
        # Check ratings
        if self.min_rating or self.max_rating:
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except (ValueError, TypeError):
                white_elo, black_elo = 0, 0
            
            if self.require_both_ratings and (white_elo == 0 or black_elo == 0):
                return False
            
            avg_elo = (white_elo + black_elo) / 2 if white_elo and black_elo else max(white_elo, black_elo)
            
            if self.min_rating and avg_elo < self.min_rating:
                return False
            if self.max_rating and avg_elo > self.max_rating:
                return False
        
        # Check time control
        if self._tc_regex:
            time_control = headers.get("TimeControl", "")
            if not self._tc_regex.match(time_control):
                return False
        
        return True


# ============================================================================
# Streaming PGN reader
# ============================================================================

class ZstdPGNReader:
    """
    Streaming reader for .pgn.zst files.
    
    Uses zstandard streaming decompression to avoid loading
    the entire file into memory.
    """
    
    def __init__(self, filepath: str, buffer_size: int = 65536):
        self.filepath = filepath
        self.buffer_size = buffer_size
    
    def iter_games(self) -> Iterator[chess.pgn.Game]:
        """Iterate over all games in the file."""
        dctx = zstd.ZstdDecompressor()
        
        with open(self.filepath, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    yield game


# ============================================================================
# Dataset
# ============================================================================

class ChessIterableDataset(IterableDataset):
    """
    Streaming dataset that yields (board_tensor, policy_target, value_target).
    
    Features:
    - Streaming from .pgn.zst files
    - Filtering games by rating, plies, time control
    - Subsampling positions within games
    - Multi-worker support
    """
    
    def __init__(
        self,
        pgn_files: List[str],
        game_filter: Optional[GameFilter] = None,
        subsample_every_k: int = 1,
        include_legal_mask: bool = False,
        shuffle_games: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            pgn_files: List of paths to .pgn.zst files
            game_filter: Filter for games
            subsample_every_k: Only yield every K-th position
            include_legal_mask: If True, also yield legal move mask
            shuffle_games: Shuffle file order (per worker)
            seed: Random seed for shuffling
        """
        super().__init__()
        self.pgn_files = pgn_files
        self.game_filter = game_filter or GameFilter()
        self.subsample_every_k = subsample_every_k
        self.include_legal_mask = include_legal_mask
        self.shuffle_games = shuffle_games
        self.seed = seed
    
    def _get_game_value(self, result: str, turn: chess.Color) -> float:
        """
        Convert game result to value target.
        
        Value is from the perspective of the player to move.
        """
        if result == "1-0":
            return 1.0 if turn == chess.WHITE else -1.0
        elif result == "0-1":
            return -1.0 if turn == chess.WHITE else 1.0
        else:  # Draw or unknown
            return 0.0
    
    def _process_game(
        self, 
        game: chess.pgn.Game
    ) -> Iterator[Tuple]:
        """
        Process a single game and yield training samples.
        """
        result = game.headers.get("Result", "*")
        board = game.board()
        
        moves = list(game.mainline_moves())
        
        for ply, move in enumerate(moves):
            # Subsample positions
            if ply % self.subsample_every_k != 0:
                board.push(move)
                continue
            
            # Skip if no legal moves (shouldn't happen in valid games)
            if not list(board.legal_moves):
                board.push(move)
                continue
            
            try:
                # Encode board
                board_tensor = board_to_tensor_fast(board)
                
                # Policy target (one-hot)
                move_idx = move_to_index(move, board.turn)
                policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
                policy_target[move_idx] = 1.0
                
                # Value target
                value_target = self._get_game_value(result, board.turn)
                
                if self.include_legal_mask:
                    legal_mask = get_legal_mask(board)
                    yield board_tensor, policy_target, value_target, legal_mask
                else:
                    yield board_tensor, policy_target, value_target
                    
            except (ValueError, KeyError):
                # Skip positions with encoding issues
                pass
            
            board.push(move)
    
    def _get_worker_files(self) -> List[str]:
        """Get files assigned to this worker."""
        worker_info = get_worker_info()
        
        files = list(self.pgn_files)
        
        if self.shuffle_games:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(files)
        
        if worker_info is not None:
            # Partition files among workers
            per_worker = len(files) // worker_info.num_workers
            remainder = len(files) % worker_info.num_workers
            
            start = worker_info.id * per_worker + min(worker_info.id, remainder)
            end = start + per_worker + (1 if worker_info.id < remainder else 0)
            
            files = files[start:end]
        
        return files
    
    def __iter__(self) -> Iterator[Tuple]:
        """Iterate over all positions in all files."""
        files = self._get_worker_files()
        
        for filepath in files:
            reader = ZstdPGNReader(filepath)
            
            for game in reader.iter_games():
                # Apply filter
                if not self.game_filter.accept_game(game):
                    continue
                
                # Yield positions from this game
                yield from self._process_game(game)


# ============================================================================
# DataLoader utilities
# ============================================================================

def worker_init_fn(worker_id: int):
    """
    Initialize worker with unique random seed.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_dataloader(
    pgn_files: List[str],
    batch_size: int = 256,
    num_workers: int = 4,
    game_filter: Optional[GameFilter] = None,
    subsample_every_k: int = 1,
    include_legal_mask: bool = False,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create an optimized DataLoader for chess training.
    """
    dataset = ChessIterableDataset(
        pgn_files=pgn_files,
        game_filter=game_filter,
        subsample_every_k=subsample_every_k,
        include_legal_mask=include_legal_mask,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )


# ============================================================================
# Testing
# ============================================================================

def test_dataset_basic(pgn_path: str, num_samples: int = 10):
    """
    Basic test to verify dataset produces valid tensors.
    """
    print(f"Testing dataset with {pgn_path}...")
    
    dataset = ChessIterableDataset(
        pgn_files=[pgn_path],
        include_legal_mask=True,
    )
    
    count = 0
    for sample in dataset:
        board_tensor, policy, value, mask = sample
        
        # Check shapes
        assert board_tensor.shape == (18, 8, 8), f"Bad board shape: {board_tensor.shape}"
        assert policy.shape == (4672,), f"Bad policy shape: {policy.shape}"
        assert isinstance(value, float), f"Bad value type: {type(value)}"
        assert mask.shape == (4672,), f"Bad mask shape: {mask.shape}"
        
        # Check values
        assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
        assert np.sum(policy) == 1.0, f"Policy not one-hot: {np.sum(policy)}"
        assert np.sum(mask) > 0, "Mask has no legal moves"
        
        # Policy move should be legal
        assert mask[np.argmax(policy)] == 1.0, "Policy move not legal"
        
        count += 1
        if count >= num_samples:
            break
    
    print(f"✓ Tested {count} samples, all valid!")
    return True


def test_dataloader(pgn_path: str, batch_size: int = 32):
    """
    Test DataLoader produces valid batches.
    """
    print(f"Testing DataLoader with batch_size={batch_size}...")
    
    loader = create_dataloader(
        pgn_files=[pgn_path],
        batch_size=batch_size,
        num_workers=2,
        include_legal_mask=True,
    )
    
    start_time = time.time()
    num_batches = 0
    total_samples = 0
    
    for batch in loader:
        boards, policies, values, masks = batch
        
        assert boards.shape == (batch_size, 18, 8, 8)
        assert policies.shape == (batch_size, 4672)
        assert values.shape == (batch_size,)
        assert masks.shape == (batch_size, 4672)
        
        num_batches += 1
        total_samples += batch_size
        
        if num_batches >= 10:
            break
    
    elapsed = time.time() - start_time
    throughput = total_samples / elapsed
    
    print(f"✓ {num_batches} batches processed")
    print(f"✓ Throughput: {throughput:.1f} samples/sec")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pgn_path = sys.argv[1]
        test_dataset_basic(pgn_path)
        print()
        test_dataloader(pgn_path)
    else:
        print("Usage: python data.py <path_to_pgn.zst>")
