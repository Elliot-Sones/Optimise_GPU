#!/usr/bin/env python3
"""
Create a validation split from a large .pgn.zst file.
Extracts the first N games to a new file.
"""

import sys
import argparse
import zstandard as zstd
import io
import chess.pgn

def create_split(input_path: str, output_path: str, num_games: int):
    print(f"Reading from: {input_path}")
    print(f"Writing first {num_games} games to: {output_path}")
    
    # Readers
    dctx = zstd.ZstdDecompressor()
    
    # Writers
    cctx = zstd.ZstdCompressor()
    
    count = 0
    
    with open(input_path, 'rb') as fh_in, \
         open(output_path, 'wb') as fh_out:
        
        with dctx.stream_reader(fh_in) as reader, \
             cctx.stream_writer(fh_out) as writer:
            
            text_in = io.TextIOWrapper(reader, encoding='utf-8')
            text_out = io.TextIOWrapper(writer, encoding='utf-8')
            
            while count < num_games:
                game = chess.pgn.read_game(text_in)
                if game is None:
                    break
                
                print(game, file=text_out, end="\n\n")
                count += 1
                
                if count % 100 == 0:
                    print(f"Extracted {count} games...", end='\r')
            
            text_out.flush()
    
    print(f"\nDone! Saved {count} games to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 make_val_split.py <input.pgn.zst> <output.pgn.zst> [num_games]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    
    create_split(input_file, output_file, num)
