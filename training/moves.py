"""
AlphaZero-style move encoding for chess.

Encodes moves as indices in [0, 4671] representing 8x8x73 action planes:
- 56 queen-type moves (8 directions × 7 distances)
- 8 knight moves
- 9 underpromotions (3 piece types × 3 directions)

All moves are encoded from the perspective of the current player (board is
flipped for black before encoding).
"""

import chess
import numpy as np
from typing import Optional, Tuple, List

# ============================================================================
# Constants
# ============================================================================

NUM_SQUARES = 64
NUM_PLANES = 73
NUM_ACTIONS = NUM_SQUARES * NUM_PLANES  # 4672

# Direction vectors for queen moves (N, NE, E, SE, S, SW, W, NW)
QUEEN_DIRECTIONS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight move offsets
KNIGHT_MOVES = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Underpromotion pieces (queen promotion is encoded as regular move)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion directions: left-capture, forward, right-capture
UNDERPROMOTION_DIRECTIONS = [-1, 0, 1]


# ============================================================================
# Encoding helpers
# ============================================================================

def _square_to_coords(square: int) -> Tuple[int, int]:
    """Convert chess square (0-63) to (file, rank) coordinates."""
    return chess.square_file(square), chess.square_rank(square)


def _coords_to_square(file: int, rank: int) -> int:
    """Convert (file, rank) to chess square."""
    return chess.square(file, rank)


def _flip_square(square: int) -> int:
    """Flip square vertically (for black's perspective)."""
    file, rank = _square_to_coords(square)
    return _coords_to_square(file, 7 - rank)


def _get_queen_move_plane(from_sq: int, to_sq: int) -> Optional[int]:
    """
    Get plane index (0-55) for a queen-type move.
    Returns None if not a valid queen move.
    """
    from_file, from_rank = _square_to_coords(from_sq)
    to_file, to_rank = _square_to_coords(to_sq)
    
    df = to_file - from_file
    dr = to_rank - from_rank
    
    # Determine direction and distance
    if df == 0 and dr > 0:
        direction = 0  # N
        distance = dr
    elif df > 0 and dr > 0 and df == dr:
        direction = 1  # NE
        distance = df
    elif df > 0 and dr == 0:
        direction = 2  # E
        distance = df
    elif df > 0 and dr < 0 and df == -dr:
        direction = 3  # SE
        distance = df
    elif df == 0 and dr < 0:
        direction = 4  # S
        distance = -dr
    elif df < 0 and dr < 0 and df == dr:
        direction = 5  # SW
        distance = -df
    elif df < 0 and dr == 0:
        direction = 6  # W
        distance = -df
    elif df < 0 and dr > 0 and -df == dr:
        direction = 7  # NW
        distance = -df
    else:
        return None  # Not a queen move
    
    if distance < 1 or distance > 7:
        return None
    
    return direction * 7 + (distance - 1)


def _get_knight_move_plane(from_sq: int, to_sq: int) -> Optional[int]:
    """
    Get plane index (56-63) for a knight move.
    Returns None if not a valid knight move.
    """
    from_file, from_rank = _square_to_coords(from_sq)
    to_file, to_rank = _square_to_coords(to_sq)
    
    df = to_file - from_file
    dr = to_rank - from_rank
    
    for i, (kdf, kdr) in enumerate(KNIGHT_MOVES):
        if df == kdf and dr == kdr:
            return 56 + i
    
    return None


def _get_underpromotion_plane(from_sq: int, to_sq: int, 
                               promotion: int) -> Optional[int]:
    """
    Get plane index (64-72) for an underpromotion move.
    Returns None if not a valid underpromotion.
    """
    if promotion not in UNDERPROMOTION_PIECES:
        return None
    
    from_file, from_rank = _square_to_coords(from_sq)
    to_file, to_rank = _square_to_coords(to_sq)
    
    # Must be a promotion (reaching rank 7 from white's perspective)
    if to_rank != 7:
        return None
    
    df = to_file - from_file
    if df not in UNDERPROMOTION_DIRECTIONS:
        return None
    
    piece_idx = UNDERPROMOTION_PIECES.index(promotion)
    dir_idx = UNDERPROMOTION_DIRECTIONS.index(df)
    
    return 64 + piece_idx * 3 + dir_idx


# ============================================================================
# Main encoding/decoding functions
# ============================================================================

def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """
    Convert a chess move to an action index in [0, 4671].
    
    Args:
        move: The chess move to encode
        turn: Current player's turn (chess.WHITE or chess.BLACK)
    
    Returns:
        Action index in [0, 4671]
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promotion = move.promotion
    
    # Flip perspective for black
    if turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
        to_sq = _flip_square(to_sq)
    
    # Try underpromotion first (most specific)
    if promotion is not None and promotion != chess.QUEEN:
        plane = _get_underpromotion_plane(from_sq, to_sq, promotion)
        if plane is not None:
            return from_sq * NUM_PLANES + plane
    
    # Try knight move
    plane = _get_knight_move_plane(from_sq, to_sq)
    if plane is not None:
        return from_sq * NUM_PLANES + plane
    
    # Queen move (includes queen promotions)
    plane = _get_queen_move_plane(from_sq, to_sq)
    if plane is not None:
        return from_sq * NUM_PLANES + plane
    
    raise ValueError(f"Cannot encode move {move}")


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """
    Convert an action index back to a chess move.
    
    Args:
        idx: Action index in [0, 4671]
        board: Current board state (needed for legality and promotions)
    
    Returns:
        The corresponding chess.Move
    
    Raises:
        ValueError: If the index doesn't correspond to a legal move
    """
    from_sq = idx // NUM_PLANES
    plane = idx % NUM_PLANES
    turn = board.turn
    
    # Flip back for black
    if turn == chess.BLACK:
        from_sq = _flip_square(from_sq)
    
    # Decode the plane
    if plane < 56:
        # Queen move
        direction = plane // 7
        distance = (plane % 7) + 1
        df, dr = QUEEN_DIRECTIONS[direction]
        df *= distance
        dr *= distance
        
        from_file, from_rank = _square_to_coords(from_sq)
        to_file = from_file + (df if turn == chess.WHITE else -df)
        to_rank = from_rank + (dr if turn == chess.WHITE else -dr)
        
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            raise ValueError(f"Invalid move index {idx}")
        
        to_sq = _coords_to_square(to_file, to_rank)
        
        # Check for pawn promotion
        piece = board.piece_at(from_sq)
        promotion = None
        if piece and piece.piece_type == chess.PAWN:
            if (turn == chess.WHITE and to_rank == 7) or \
               (turn == chess.BLACK and to_rank == 0):
                promotion = chess.QUEEN
        
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        
    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        kdf, kdr = KNIGHT_MOVES[knight_idx]
        
        from_file, from_rank = _square_to_coords(from_sq)
        if turn == chess.BLACK:
            kdf = -kdf
            kdr = -kdr
        
        to_file = from_file + kdf
        to_rank = from_rank + kdr
        
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            raise ValueError(f"Invalid move index {idx}")
        
        to_sq = _coords_to_square(to_file, to_rank)
        move = chess.Move(from_sq, to_sq)
        
    else:
        # Underpromotion
        underpromo_idx = plane - 64
        piece_idx = underpromo_idx // 3
        dir_idx = underpromo_idx % 3
        
        promotion = UNDERPROMOTION_PIECES[piece_idx]
        df = UNDERPROMOTION_DIRECTIONS[dir_idx]
        
        from_file, from_rank = _square_to_coords(from_sq)
        if turn == chess.BLACK:
            df = -df
        
        to_file = from_file + df
        to_rank = 7 if turn == chess.WHITE else 0
        
        if not (0 <= to_file <= 7):
            raise ValueError(f"Invalid move index {idx}")
        
        to_sq = _coords_to_square(to_file, to_rank)
        move = chess.Move(from_sq, to_sq, promotion=promotion)
    
    # Validate the move is legal
    if move not in board.legal_moves:
        raise ValueError(f"Move {move} from index {idx} is not legal")
    
    return move


def get_legal_mask(board: chess.Board) -> np.ndarray:
    """
    Get a binary mask of legal moves.
    
    Args:
        board: Current board state
    
    Returns:
        numpy array of shape (4672,) with 1s for legal moves
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board.turn)
            mask[idx] = 1.0
        except ValueError:
            # Skip moves that can't be encoded (shouldn't happen)
            pass
    
    return mask


def get_legal_move_indices(board: chess.Board) -> List[int]:
    """
    Get list of action indices for all legal moves.
    
    Args:
        board: Current board state
    
    Returns:
        List of action indices
    """
    indices = []
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board.turn)
            indices.append(idx)
        except ValueError:
            pass
    return indices


# ============================================================================
# Sanity tests
# ============================================================================

def test_move_encoding_roundtrip(num_games: int = 100, verbose: bool = False):
    """
    Test that all legal moves in random games encode and decode correctly.
    """
    import random
    
    errors = 0
    total_moves = 0
    
    for game_idx in range(num_games):
        board = chess.Board()
        
        # Play random moves
        for _ in range(random.randint(10, 100)):
            if board.is_game_over():
                break
            
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            
            # Test encoding
            try:
                idx = move_to_index(move, board.turn)
                decoded = index_to_move(idx, board)
                
                if decoded != move:
                    if verbose:
                        print(f"Mismatch: {move} -> {idx} -> {decoded}")
                    errors += 1
            except ValueError as e:
                if verbose:
                    print(f"Error encoding {move}: {e}")
                errors += 1
            
            total_moves += 1
            board.push(move)
    
    print(f"Move encoding round-trip test: {total_moves} moves, {errors} errors")
    if errors == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {errors} errors found")
    
    return errors == 0


def test_legal_mask(num_positions: int = 100):
    """
    Test that legal masks correctly identify all legal moves.
    """
    import random
    
    errors = 0
    
    for _ in range(num_positions):
        board = chess.Board()
        
        # Play random moves to get varied positions
        for _ in range(random.randint(0, 50)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if moves:
                board.push(random.choice(moves))
        
        if board.is_game_over():
            continue
        
        mask = get_legal_mask(board)
        legal_indices = set(get_legal_move_indices(board))
        mask_indices = set(np.where(mask > 0)[0])
        
        if legal_indices != mask_indices:
            print(f"Mask mismatch at position: {board.fen()}")
            errors += 1
    
    print(f"Legal mask test: {num_positions} positions, {errors} errors")
    if errors == 0:
        print("✓ All tests passed!")
    
    return errors == 0


if __name__ == "__main__":
    print("Running move encoding tests...\n")
    test_move_encoding_roundtrip(100)
    print()
    test_legal_mask(100)
