#!/usr/bin/env python3
"""
Analyze training data to check for:
1. Right-to-left placement bias
2. Multiple rotations per piece
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_training_data(data_dir: str = "./data/training_data_clean"):
    """Analyze training data for placement bias and rotation diversity."""
    
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} training data files\n")
    
    # Statistics
    total_moves = 0
    col_distribution = defaultdict(int)  # Count placements by column
    rotation_by_piece = defaultdict(lambda: defaultdict(int))  # piece -> rotation -> count
    piece_counts = defaultdict(int)
    
    # Track placement patterns
    right_side_placements = 0  # col >= 7 (right side of 10-column board)
    left_side_placements = 0   # col <= 2 (left side)
    center_placements = 0      # 3 <= col <= 6
    
    # Track sequences for right-to-left movement
    right_to_left_sequences = 0
    left_to_right_sequences = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
            
            moves = game_data.get("moves", [])
            if not moves:
                continue
            
            prev_col = None
            for move in moves:
                if move.get("move_type") != "place":
                    continue
                
                piece_type = move.get("piece_type", "").upper()
                piece_position = move.get("piece_position", {})
                col = piece_position.get("col", 0)
                rotation = piece_position.get("rotation", 0)
                
                total_moves += 1
                col_distribution[col] += 1
                rotation_by_piece[piece_type][rotation] += 1
                piece_counts[piece_type] += 1
                
                # Track side placements
                if col >= 7:
                    right_side_placements += 1
                elif col <= 2:
                    left_side_placements += 1
                else:
                    center_placements += 1
                
                # Track sequential movement direction
                if prev_col is not None:
                    if col < prev_col:
                        right_to_left_sequences += 1
                    elif col > prev_col:
                        left_to_right_sequences += 1
                
                prev_col = col
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Print results
    print("=" * 80)
    print("PLACEMENT BIAS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal moves analyzed: {total_moves:,}")
    print(f"\nColumn distribution:")
    for col in sorted(col_distribution.keys()):
        count = col_distribution[col]
        percentage = (count / total_moves * 100) if total_moves > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"  Col {col:2d}: {count:6,} ({percentage:5.1f}%) {bar}")
    
    print(f"\nSide placement distribution:")
    print(f"  Left side (col 0-2):   {left_side_placements:6,} ({left_side_placements/total_moves*100:5.1f}%)")
    print(f"  Center (col 3-6):      {center_placements:6,} ({center_placements/total_moves*100:5.1f}%)")
    print(f"  Right side (col 7-9):  {right_side_placements:6,} ({right_side_placements/total_moves*100:5.1f}%)")
    
    print(f"\nSequential movement direction:")
    print(f"  Left-to-right:  {left_to_right_sequences:6,}")
    print(f"  Right-to-left:  {right_to_left_sequences:6,}")
    
    if left_to_right_sequences > 0 or right_to_left_sequences > 0:
        ratio = left_to_right_sequences / right_to_left_sequences if right_to_left_sequences > 0 else float('inf')
        print(f"  Ratio (L→R / R→L): {ratio:.2f}")
        if ratio > 2.0:
            print("  ⚠️  WARNING: Strong left-to-right bias detected!")
        elif ratio < 0.5:
            print("  ⚠️  WARNING: Strong right-to-left bias detected!")
        else:
            print("  ✓ Balanced movement direction")
    
    print("\n" + "=" * 80)
    print("ROTATION DIVERSITY ANALYSIS")
    print("=" * 80)
    
    for piece_type in sorted(piece_counts.keys()):
        print(f"\n{piece_type} piece (total: {piece_counts[piece_type]:,} placements):")
        rotations = rotation_by_piece[piece_type]
        total_for_piece = sum(rotations.values())
        
        for rot in sorted(rotations.keys()):
            count = rotations[rot]
            percentage = (count / total_for_piece * 100) if total_for_piece > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"  Rotation {rot}: {count:6,} ({percentage:5.1f}%) {bar}")
        
        unique_rotations = len(rotations)
        if unique_rotations < 2:
            print(f"  ⚠️  WARNING: Only {unique_rotations} rotation(s) found for {piece_type}!")
        elif unique_rotations == 4:
            print(f"  ✓ All 4 rotations present")
        else:
            print(f"  ⚠️  Only {unique_rotations} out of 4 rotations present")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Check for right-to-left placements
    has_right_to_left = right_to_left_sequences > 0
    print(f"\n1. Right-to-left placement exists: {'✓ YES' if has_right_to_left else '✗ NO'}")
    if has_right_to_left:
        print(f"   Found {right_to_left_sequences:,} right-to-left sequential movements")
    else:
        print("   ⚠️  No right-to-left sequential movements found!")
    
    # Check for multiple rotations
    all_pieces_have_multiple_rotations = True
    pieces_with_single_rotation = []
    for piece_type in sorted(piece_counts.keys()):
        unique_rotations = len(rotation_by_piece[piece_type])
        if unique_rotations < 2:
            all_pieces_have_multiple_rotations = False
            pieces_with_single_rotation.append(piece_type)
    
    print(f"\n2. Multiple rotations per piece: {'✓ YES' if all_pieces_have_multiple_rotations else '✗ NO'}")
    if all_pieces_have_multiple_rotations:
        print("   All pieces have multiple rotations")
    else:
        print(f"   ⚠️  Pieces with only 1 rotation: {', '.join(pieces_with_single_rotation)}")
    
    return {
        'total_moves': total_moves,
        'col_distribution': dict(col_distribution),
        'rotation_by_piece': {k: dict(v) for k, v in rotation_by_piece.items()},
        'has_right_to_left': has_right_to_left,
        'all_pieces_have_multiple_rotations': all_pieces_have_multiple_rotations
    }

if __name__ == "__main__":
    analyze_training_data("./data/training_data_clean")

