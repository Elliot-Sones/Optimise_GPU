import chess
import chess.pgn
import zstandard as zstd
import io
import sys
import numpy as np

# Add training dir to path
sys.path.append('training')
from data import ZstdPGNReader, board_to_tensor_fast
import moves
from moves import index_to_move

def create_sample_game():
    # Create a simple game: Scholar's Mate
    game = chess.pgn.Game()
    game.headers["Event"] = "Sample Game"
    game.headers["Site"] = "Localhost"
    game.headers["Date"] = "2024.01.01"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player A"
    game.headers["Black"] = "Player B"
    game.headers["Result"] = "1-0"
    game.headers["WhiteElo"] = "1500"
    game.headers["BlackElo"] = "1500"
    
    node = game.add_variation(chess.Move.from_uci("e2e4"))
    node = node.add_variation(chess.Move.from_uci("e7e5"))
    node = node.add_variation(chess.Move.from_uci("d1h5"))
    node = node.add_variation(chess.Move.from_uci("b8c6"))
    node = node.add_variation(chess.Move.from_uci("f1c4"))
    node = node.add_variation(chess.Move.from_uci("g8f6"))
    node = node.add_variation(chess.Move.from_uci("h5f7")) # Checkmate!
    
    return game

def save_compressed_pgn(game, filename="sample.pgn.zst"):
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)
    
    print(f"Original PGN Content:\n{'-'*40}\n{pgn_string}{'-'*40}\n")
    
    cctx = zstd.ZstdCompressor()
    with open(filename, "wb") as f:
        with cctx.stream_writer(f) as writer:
            writer.write(pgn_string.encode('utf-8'))
            
    print(f"Saved compressed game to {filename}")

def inspect_data(filename="sample.pgn.zst"):
    print(f"\nReading back from {filename} using training pipeline...\n")
    reader = ZstdPGNReader(filename)
    
    for i, game in enumerate(reader.iter_games()):
        print(f"Game {i+1}: {game.headers['White']} vs {game.headers['Black']} ({game.headers['Result']})")
        
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            # Just show the first few moves to demonstrate encoding
            if ply > 2: 
                break
                
            print(f"\n--- Ply {ply} (Turn: {'White' if board.turn else 'Black'}) ---")
            print(f"Move played: {move}")
            
            # encode
            tensor = board_to_tensor_fast(board)
            print(f"Input Tensor Shape: {tensor.shape} (18 planes x 8x8)")
            
            # Visualize a few planes
            print("Plane 0 (Our Pawns):")
            print(tensor[0].astype(int))
            
            # Policy target
            move_idx = moves.move_to_index(move, board.turn)
            print(f"Target Move Index: {move_idx}")
            
            # Decode back to verify
            decoded_move = moves.index_to_move(move_idx, board)
            print(f"Decoded back: {decoded_move}")
            
            board.push(move)

if __name__ == "__main__":
    game = create_sample_game()
    save_compressed_pgn(game)
    inspect_data()
