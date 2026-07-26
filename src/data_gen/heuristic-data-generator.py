"""Heuristic data generator for puzzle grid training data.

This script plays puzzle grid games using heuristics to generate successful
placement data for reinforcement learning training.
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement


BASE_URL = "http://localhost:8000"


class HeuristicPlayer:
    """Plays puzzle grid games using heuristics to generate training data."""
    
    def __init__(self, piece_types: List[str], api_url: str = BASE_URL, max_lines: int = 1000, placement_direction: str = "left-to-right", preferred_rotation: Optional[int] = None):
        """Initialize the heuristic player.
        
        Args:
            piece_types: List of piece types to use (e.g., ["I"], ["I", "L"], ["I", "O", "T", "S", "Z", "J", "L"])
                         Must contain 1 to 7 piece types
            api_url: Base URL of the PuzzleGrid API
            max_lines: Maximum number of lines to clear before stopping (default: 1000)
            placement_direction: Direction to favor when placing pieces. "left-to-right" (default) or "right-to-left"
            preferred_rotation: If provided (0-3), prefer placements with this rotation when multiple valid options exist.
                               Only used for single-piece games. None means no preference.
        """
        if not piece_types or len(piece_types) == 0:
            raise ValueError("At least one piece type must be provided")
        if len(piece_types) > 7:
            raise ValueError("Maximum of 7 piece types allowed (I, O, T, S, Z, J, L)")
        if placement_direction not in ["left-to-right", "right-to-left"]:
            raise ValueError("placement_direction must be 'left-to-right' or 'right-to-left'")
        if preferred_rotation is not None and (preferred_rotation < 0 or preferred_rotation > 3):
            raise ValueError("preferred_rotation must be between 0 and 3, or None")
        
        # Normalize and validate piece types
        self.piece_types = [pt.upper() for pt in piece_types]
        self.api_url = api_url
        self.game_id = None
        self.max_lines = max_lines
        self.placement_direction = placement_direction
        self.preferred_rotation = preferred_rotation  # Only used for single-piece games
        
        # Validate all piece types
        for piece_type in self.piece_types:
            try:
                PieceType[piece_type]
            except KeyError:
                raise ValueError(f"Invalid piece type: {piece_type}. Valid types: {[p.name for p in PieceType]}")
    
    def create_game(self) -> str:
        """Create a new game with the specified piece types.
        
        Returns:
            Game ID
        """
        response = requests.post(
            f"{self.api_url}/games",
            json={
                "width": 10,
                "height": 20,
                "allowed_pieces": self.piece_types,
            }
        )
        response.raise_for_status()
        game_data = response.json()
        self.game_id = game_data["game_id"]
        piece_types_str = ", ".join(self.piece_types)
        print(f"Created game {self.game_id} with piece types: {piece_types_str}")
        return self.game_id
    
    def get_game_state(self) -> Dict:
        """Get the current game state.
        
        Returns:
            Game state dictionary
        """
        response = requests.get(f"{self.api_url}/games/{self.game_id}")
        response.raise_for_status()
        return response.json()
    
    def place_piece(self, row: int, col: int, rotation: int) -> bool:
        """Place the current piece at the specified position.
        
        Args:
            row: Row position
            col: Column position
            rotation: Rotation state (0-3)
            
        Returns:
            True if placement was successful
        """
        response = requests.post(
            f"{self.api_url}/games/{self.game_id}/place",
            json={
                "row": row,
                "col": col,
                "rotation": rotation,
            }
        )
        if response.status_code != 200:
            # Log the error for debugging
            try:
                error_detail = response.json().get("detail", "Unknown error")
                if "floating" not in str(error_detail).lower() and "rest" not in str(error_detail).lower():
                    # Only log if it's not a floating piece error (we should catch those)
                    pass
            except:
                pass
        return response.status_code == 200
    
    def get_valid_placements(self, board: List[List[int]], piece_type: PieceType) -> List[Tuple[int, int, int]]:
        """Get all valid placements for a piece type on the board.
        
        Args:
            board: Current board state
            piece_type: Type of piece to place
            
        Returns:
            List of (row, col, rotation) tuples representing valid placements
        """
        # Canonical reachable resting placements (BFS from spawn). This replaces the
        # old per-column landing scan, which had no reachability check and could emit
        # placements inside enclosed/unreachable pockets into the training data.
        valid_placements = placement.get_valid_placements(board, piece_type)

        # Preserve the placement_direction bias as a tie-break ordering: left-to-right
        # sorts by ascending column, right-to-left by descending column.
        reverse = self.placement_direction == "right-to-left"
        valid_placements.sort(key=lambda p: (p[1], p[0], p[2]), reverse=reverse)

        return valid_placements
    
    def evaluate_placement(
        self,
        board: List[List[int]],
        row: int,
        col: int,
        rotation: int,
        piece_type: PieceType
    ) -> float:
        """Evaluate a placement using heuristics.
        
        Args:
            board: Current board state
            row: Row position
            col: Column position
            rotation: Rotation state
            piece_type: Type of piece
            
        Returns:
            Score for this placement (higher is better)
        """
        # I-piece specific evaluation - prioritize row completion
        if piece_type == PieceType.I:
            return self._evaluate_i_piece_placement(board, row, col, rotation)
        
        # Create a copy of the board to simulate placement
        test_board = [row[:] for row in board]
        width = len(board[0])
        height = len(board)
        
        # Place the piece
        temp_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
        cells = temp_piece.get_cells()
        
        # Determine if piece is horizontal or vertical (needed for I-piece logic)
        piece_rows = set(r for r, _ in cells)
        piece_cols = set(c for _, c in cells)
        is_horizontal = len(piece_rows) < len(piece_cols)
        
        # Mark cells as placed (use 1 as placeholder)
        for r, c in cells:
            if 0 <= r < height and 0 <= c < width:
                test_board[r][c] = 1
        
        # Calculate heuristics
        score = 0.0
        
        # 0. Check board height - if board is getting tall, prioritize line clearing even more
        max_height = 0
        for c in range(width):
            for r in range(height):
                if board[r][c] != 0:
                    max_height = max(max_height, height - r)
                    break
        
        # If board is getting tall, we need to clear lines urgently
        height_urgency = 0.0
        if max_height > 15:
            height_urgency = 1000.0  # Urgent - board is very tall
        elif max_height > 12:
            height_urgency = 500.0
        elif max_height > 10:
            height_urgency = 200.0
        
        # 1. Lines cleared (highest priority - MASSIVE bonus)
        lines_cleared = 0
        cleared_rows = []
        for r in range(height):
            if all(cell != 0 for cell in test_board[r]):
                lines_cleared += 1
                cleared_rows.append(r)
        score += lines_cleared * (10000.0 + height_urgency)  # Massive bonus, increased if board is tall
        
        # Bonus for clearing multiple lines at once
        if lines_cleared >= 4:
            score += 50000.0  # Huge bonus for clearing 4 lines
        elif lines_cleared >= 2:
            score += lines_cleared * 2000.0  # Bonus for multiple lines
        
        # If board is tall and we're not clearing lines, this is bad
        if height_urgency > 0 and lines_cleared == 0:
            score -= height_urgency * 2.0  # Strong penalty for not clearing when board is tall
        
        # 2. Row completion potential (NEW - prioritize completing partial rows)
        # Check how many cells each row has filled, focusing on lower rows first
        row_completion_bonus = 0.0
        rows_near_completion = 0
        rows_this_piece_helps = set()
        
        # Check which rows this piece contributes to
        piece_contributed_rows = set(r for r, _ in cells if 0 <= r < height)
        
        for r in range(height - 1, max(0, height - 15), -1):  # Check bottom 15 rows
            filled_cells = sum(1 for cell in test_board[r] if cell != 0)
            if filled_cells > 0 and filled_cells < width:
                # Partial row - bonus for getting closer to completion
                completion_ratio = filled_cells / width
                # Much higher bonus for rows very close to completion
                # Also higher bonus for lower rows (easier to complete)
                row_weight = (height - r) / height  # Lower rows have higher weight
                
                # Extra bonus if this piece contributes to this row
                piece_helps_this_row = r in piece_contributed_rows
                contribution_multiplier = 2.0 if piece_helps_this_row else 1.0
                
                if filled_cells >= 8:
                    bonus = 500.0 * row_weight * contribution_multiplier
                    row_completion_bonus += bonus
                    rows_near_completion += 1
                    if piece_helps_this_row:
                        rows_this_piece_helps.add(r)
                elif filled_cells >= 7:
                    bonus = 300.0 * row_weight * contribution_multiplier
                    row_completion_bonus += bonus
                    rows_near_completion += 1
                    if piece_helps_this_row:
                        rows_this_piece_helps.add(r)
                elif filled_cells >= 6:
                    row_completion_bonus += 150.0 * row_weight * contribution_multiplier
                    if piece_helps_this_row:
                        rows_this_piece_helps.add(r)
                elif filled_cells >= 5:
                    row_completion_bonus += 80.0 * row_weight * contribution_multiplier
                    if piece_helps_this_row:
                        rows_this_piece_helps.add(r)
                else:
                    row_completion_bonus += completion_ratio * 30.0 * row_weight * contribution_multiplier
            elif filled_cells == width:
                # Full row (should be cleared, but if not, huge penalty)
                row_completion_bonus -= 1000.0
        
        # Extra bonus if this placement helps complete multiple rows
        score += row_completion_bonus
        if rows_near_completion > 1:
            score += rows_near_completion * 100.0
        
        # For I-piece: huge bonus if it completes a row horizontally
        if piece_type == PieceType.I and is_horizontal:
            for r in rows_this_piece_helps:
                filled = sum(1 for cell in test_board[r] if cell != 0)
                completion_bonus = 0.0
                if filled >= 8:  # Very close to completion
                    completion_bonus = 2000.0 + height_urgency  # Massive bonus, more if board is tall
                elif filled >= 6:
                    completion_bonus = 500.0 + height_urgency * 0.5
                elif filled >= 4:
                    completion_bonus = 200.0
                score += completion_bonus
        
        # 3. Penalize tall columns (NEW - discourage vertical stacking)
        max_column_height = 0
        column_heights = []
        for c in range(width):
            col_height = 0
            for r in range(height):
                if test_board[r][c] != 0:
                    col_height = height - r
                    break
            column_heights.append(col_height)
            max_column_height = max(max_column_height, col_height)
        
        # Penalize if any column is getting too tall
        if max_column_height > 10:
            score -= (max_column_height - 10) * 10.0  # Increased penalty
        
        # For I-piece: VERY strong penalty for building tall columns (max 12 blocks = 3 I pieces)
        if piece_type == PieceType.I:
            # I-piece specific limit: 12 blocks (3 I pieces stacked vertically)
            max_i_piece_height = 12
            if max_column_height > max_i_piece_height:
                score -= (max_column_height - max_i_piece_height) * 10000.0  # Massive penalty for exceeding limit
            
            if not is_horizontal:
                # Vertical I-piece - VERY strong penalty unless it clears lines
                if lines_cleared == 0:
                    score -= 2000.0  # Increased from 500.0 - very strong penalty for vertical I that doesn't clear
                # Also check column height - if it would exceed 3 I-pieces (12 blocks), huge penalty
                for c in piece_cols:
                    current_col_height = 0
                    for r in range(height):
                        if board[r][c] != 0:
                            current_col_height = height - r
                            break
                    if current_col_height + 4 > max_i_piece_height:  # Would exceed 3 I-pieces
                        score -= 50000.0  # Massive penalty - should never happen due to filtering, but safety check
                avg_height = sum(column_heights) / len(column_heights) if column_heights else 0
                if max_column_height > avg_height + 1:
                    score -= 500.0  # Increased from 300.0 - strong penalty for making columns taller
            else:
                # Horizontal I-piece - check if this placement would exceed 12 block limit
                for c in piece_cols:
                    if 0 <= c < width:
                        col_height_after = column_heights[c] if c < len(column_heights) else 0
                        if col_height_after > max_i_piece_height:
                            score -= 100000.0  # Massive penalty to prevent this placement
                # Small penalty if board is tall and this doesn't clear
                if max_column_height > 10 and lines_cleared == 0:
                    score -= 50.0  # Small penalty, but horizontal is still much better than vertical
        
        # 4. Aggregate height (lower is better)
        aggregate_height = 0
        for c in range(width):
            for r in range(height):
                if test_board[r][c] != 0:
                    aggregate_height += (height - r)
                    break
        score -= aggregate_height * 0.5
        
        # 5. Holes (fewer is better)
        holes = 0
        for c in range(width):
            found_block = False
            for r in range(height):
                if test_board[r][c] != 0:
                    found_block = True
                elif found_block and test_board[r][c] == 0:
                    holes += 1
        score -= holes * 10.0
        
        # 6. Bumpiness (smoothness, lower is better)
        heights = []
        for c in range(width):
            for r in range(height):
                if test_board[r][c] != 0:
                    heights.append(height - r)
                    break
            else:
                heights.append(0)
        
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        score -= bumpiness * 0.2
        
        # 7. Prefer lower placements (slightly)
        score += (height - row) * 0.1
        
        # 8. Prefer horizontal placements over vertical (NEW)
        # (is_horizontal already calculated above)
        if is_horizontal:
            # More horizontal than vertical - strongly prefer for row completion
            score += 100.0  # Increased from 50.0
            # For I-piece specifically, horizontal is MUCH better
            if piece_type == PieceType.I:
                score += 1000.0  # Increased from 500.0 - huge bonus for horizontal I-piece
                # Check if this horizontal I-piece can complete a row
                piece_row = list(piece_rows)[0] if piece_rows else -1
                if 0 <= piece_row < height:
                    row_filled = sum(1 for cell in test_board[piece_row] if cell != 0)
                    if row_filled >= 7:
                        score += 2000.0  # Increased from 1000.0 - massive bonus if it completes a nearly-full row
                    elif row_filled >= 6:
                        score += 500.0  # Good bonus for getting closer
        else:
            # Vertical placement - only good if it fills gaps or completes rows
            # For I-piece, vertical is usually bad unless it's filling a specific gap
            if piece_type == PieceType.I:
                # Check if this vertical I-piece is filling a gap or will help clear lines
                fills_gap = False
                will_clear_line = False
                
                # Check if it will help clear a line
                for r in piece_rows:
                    if 0 <= r < height:
                        row_filled = sum(1 for cell in test_board[r] if cell != 0)
                        if row_filled == width:  # This row will be cleared
                            will_clear_line = True
                            break
                
                # Check if it's filling a gap between existing blocks
                for r, c in cells:
                    if 0 <= r < height and 0 <= c < width:
                        # Check if there are blocks on both sides at the same row level
                        if c > 0 and c < width - 1:
                            if (r + 1 < height and board[r + 1][c - 1] != 0 and 
                                r + 1 < height and board[r + 1][c + 1] != 0):
                                fills_gap = True
                                break
                
                if will_clear_line:
                    score += 500.0  # Good if it helps clear a line
                elif not fills_gap:
                    score -= 300.0  # Strong penalty for vertical I-pieces that don't fill gaps or clear lines
        
        # 9. Prefer placements that create flat surfaces (NEW)
        # Check if placement creates a relatively flat top surface
        # This helps with future placements
        surface_heights = []
        for c in range(width):
            for r in range(height):
                if test_board[r][c] != 0:
                    surface_heights.append(height - r)
                    break
            else:
                surface_heights.append(0)
        
        if surface_heights:
            avg_height = sum(surface_heights) / len(surface_heights)
            height_variance = sum((h - avg_height) ** 2 for h in surface_heights) / len(surface_heights)
            # Lower variance = flatter surface = better
            score -= height_variance * 0.5
        
        # 10. Penalize creating isolated blocks (NEW)
        # Check if placement creates blocks that are isolated (surrounded by empty cells)
        isolated_penalty = 0
        for r, c in cells:
            if 0 <= r < height and 0 <= c < width:
                # Check neighbors
                neighbors = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if test_board[nr][nc] != 0:
                            neighbors += 1
                # If this cell has no neighbors, it's isolated
                if neighbors == 0:
                    isolated_penalty += 5.0
        score -= isolated_penalty
        
        return score
    
    def _evaluate_i_piece_placement(
        self,
        board: List[List[int]],
        row: int,
        col: int,
        rotation: int
    ) -> float:
        """I-piece specific evaluation: prioritize completing rows horizontally.
        
        Strategy:
        1. Horizontal I-pieces that complete rows get MASSIVE bonus
        2. Horizontal I-pieces that fill NEW columns in a row get high bonus
        3. Prefer lower rows (build from bottom up)
        4. Limit vertical growth - max 3 rows high (12 blocks = 3 I pieces)
        5. Only use vertical I-pieces if they clear lines or column is empty
        """
        test_board = [row[:] for row in board]
        width = len(board[0])
        height = len(board)
        
        # Place the piece
        temp_piece = Piece(PieceType.I, row=row, col=col, rotation=rotation)
        cells = temp_piece.get_cells()
        
        # Determine if horizontal or vertical
        piece_rows = set(r for r, _ in cells if 0 <= r < height)
        piece_cols = set(c for _, c in cells if 0 <= c < width)
        is_horizontal = len(piece_rows) < len(piece_cols)
        
        # Mark cells as placed
        for r, c in cells:
            if 0 <= r < height and 0 <= c < width:
                test_board[r][c] = 1
        
        score = 0.0
        
        # 1. Lines cleared (MASSIVE priority)
        lines_cleared = 0
        for r in range(height):
            if all(cell != 0 for cell in test_board[r]):
                lines_cleared += 1
        score += lines_cleared * 100000.0  # Huge bonus for clearing lines
        
        # 2. For horizontal I-pieces: prioritize row completion
        if is_horizontal:
            # Check which row this I-piece is in
            piece_row = list(piece_rows)[0] if piece_rows else -1
            if 0 <= piece_row < height:
                row_filled = sum(1 for cell in test_board[piece_row] if cell != 0)
                
                # MASSIVE bonus for completing a row (10 blocks)
                if row_filled == width:
                    score += 200000.0  # Completing a row is the ultimate goal!
                # High bonus for getting close to completion
                elif row_filled >= 8:
                    score += 10000.0  # Very close to completion
                elif row_filled >= 6:
                    score += 2000.0  # Getting there
                elif row_filled >= 4:
                    score += 500.0  # Some progress
                
                # KEY: Bonus for filling NEW columns in this row (not building towers)
                new_columns_filled = 0
                for c in piece_cols:
                    if 0 <= piece_row < height and 0 <= c < width:
                        # Check if this column was empty in this row before
                        if board[piece_row][c] == 0:
                            new_columns_filled += 1
                
                # Huge bonus for filling new columns (helps complete rows)
                score += new_columns_filled * 1000.0
                
                # Penalty for placing in columns that already have blocks in this row
                # (this prevents building towers in the same columns)
                existing_columns = 0
                for c in piece_cols:
                    if 0 <= piece_row < height and 0 <= c < width:
                        if board[piece_row][c] != 0:
                            existing_columns += 1
                score -= existing_columns * 500.0  # Penalty for building towers
                
                # Prefer lower rows (build from bottom up)
                row_bonus = (height - piece_row) * 50.0
                score += row_bonus
        
        # 3. For vertical I-pieces: only good if they clear lines or column is empty
        else:
            # Check column height
            piece_col = list(piece_cols)[0] if piece_cols else -1
            if 0 <= piece_col < width:
                current_col_height = 0
                for r in range(height):
                    if board[r][piece_col] != 0:
                        current_col_height = height - r
                        break
                
                # Only allow vertical if column is empty (0 blocks) or clears lines
                if current_col_height == 0:
                    # Empty column - OK, but prefer horizontal
                    score += 100.0
                elif lines_cleared > 0:
                    # Clears lines - OK
                    score += 10000.0
                else:
                    # Column has blocks and doesn't clear - STRONG penalty
                    score -= 20000.0
                
                # Limit vertical growth - max 3 rows high (12 blocks = 3 I-pieces)
                max_vertical_height = 12
                if current_col_height + 4 > max_vertical_height:
                    score -= 100000.0  # Much stronger penalty for exceeding height limit
        
        # 4. Penalize building tall columns (max 3 rows = 12 blocks for I pieces)
        max_column_height = 0
        column_heights_after = []
        for c in range(width):
            col_height = 0
            for r in range(height):
                if test_board[r][c] != 0:
                    col_height = height - r
                    break
            column_heights_after.append(col_height)
            max_column_height = max(max_column_height, col_height)
        
        # If any column exceeds 3 rows (12 blocks = 3 I pieces), strong penalty
        # This prevents columns from growing too tall, which would block vertical I-piece placement
        if max_column_height > 12:
            score -= (max_column_height - 12) * 5000.0  # Much stronger penalty
        
        # For horizontal I-pieces: check if this placement would cause any column to exceed 12 blocks
        if is_horizontal:
            for c in piece_cols:
                if 0 <= c < width:
                    # Check the height this column would be after placement
                    col_height_after = column_heights_after[c]
                    if col_height_after > 12:
                        # This placement would exceed the 12 block limit - STRONG penalty
                        score -= 100000.0  # Massive penalty to prevent this placement
        
        return score
    
    def find_best_placement(self, board: List[List[int]], piece_type: PieceType) -> Optional[Tuple[int, int, int]]:
        """Find the best placement for the current piece using heuristics.
        
        Args:
            board: Current board state
            piece_type: Type of piece to place
            
        Returns:
            (row, col, rotation) tuple for best placement, or None if no valid placement
        """
        valid_placements = self.get_valid_placements(board, piece_type)
        
        # Filter out any floating placements (safety check)
        filtered_placements = []
        height = len(board)
        for row, col, rotation in valid_placements:
            shape = Piece.PIECE_SHAPES[piece_type][rotation]
            cells = [(row + dr, col + dc) for dr, dc in shape]
            lowest_row = max(r for r, _ in cells) if cells else -1
            
            can_rest = False
            if lowest_row == height - 1:
                can_rest = True
            else:
                for r, c in cells:
                    if r == lowest_row and r + 1 < height:
                        if board[r + 1][c] != 0:
                            can_rest = True
                            break
            
            if can_rest:
                filtered_placements.append((row, col, rotation))
        
        if not filtered_placements:
            # No valid placements found
            return None
        
        # If we have a preferred rotation (for single-piece games), filter to prefer that rotation
        # but still allow other rotations if preferred ones don't exist or score poorly
        if self.preferred_rotation is not None and len(self.piece_types) == 1:
            # Get equivalent rotations for pieces where some rotations are the same
            preferred_rotations = [self.preferred_rotation]
            if piece_type == PieceType.I:
                if self.preferred_rotation in [0, 2]:
                    preferred_rotations = [0, 2]
                elif self.preferred_rotation in [1, 3]:
                    preferred_rotations = [1, 3]
            elif piece_type == PieceType.O:
                preferred_rotations = [0, 1, 2, 3]  # All are equivalent
            elif piece_type in [PieceType.S, PieceType.Z]:
                if self.preferred_rotation in [0, 2]:
                    preferred_rotations = [0, 2]
                elif self.preferred_rotation in [1, 3]:
                    preferred_rotations = [1, 3]
            
            # Separate placements by rotation preference
            preferred_placements = [p for p in filtered_placements if p[2] in preferred_rotations]
            other_placements = [p for p in filtered_placements if p[2] not in preferred_rotations]
            
            # Evaluate preferred placements first
            placements_to_evaluate = preferred_placements if preferred_placements else other_placements
        else:
            placements_to_evaluate = filtered_placements
        
        # Evaluate all placements
        best_placement = None
        best_score = float('-inf')
        
        if not placements_to_evaluate:
            return None
        
        for row, col, rotation in placements_to_evaluate:
            score = self.evaluate_placement(board, row, col, rotation, piece_type)
            # Strong bonus for exact preferred_rotation so we output canonical rotations (1 not 3, 0 not 2)
            if (self.preferred_rotation is not None and len(self.piece_types) == 1 and
                rotation == self.preferred_rotation):
                score += 2000.0
            if score > best_score:
                best_score = score
                best_placement = (row, col, rotation)
        
        # Final safety check: verify the selected placement is in filtered list
        if best_placement and best_placement not in filtered_placements:
            # This shouldn't happen, but if it does, use first filtered placement
            best_placement = filtered_placements[0] if filtered_placements else None
        
        return best_placement
    
    def _get_height_map(self, board: List[List[int]]) -> List[int]:
        """Get the height of each column for debugging."""
        if not board:
            return []
        width = len(board[0])
        height = len(board)
        heights = []
        for col in range(width):
            for row in range(height):
                if board[row][col] != 0:
                    heights.append(height - row)
                    break
            else:
                heights.append(0)
        return heights
    
    def play_game(self) -> Dict:
        """Play a complete game using heuristics.
        
        Returns:
            Game data dictionary with moves
        """
        piece_types_str = ", ".join(self.piece_types)
        print(f"\nStarting game with piece types: {piece_types_str}")
        print(f"Maximum lines to clear: {self.max_lines}")
        self.create_game()
        
        # Check initial state
        initial_state = self.get_game_state()
        print(f"Initial game status: {initial_state['status']}")
        print(f"Initial current piece: {initial_state.get('current_piece')}")
        
        if initial_state["status"] != "playing":
            print(f"\nGame already over at start! Status: {initial_state['status']}")
            # Still save the game data
            response = requests.get(f"{self.api_url}/games/{self.game_id}/moves")
            response.raise_for_status()
            return response.json()
        
        moves_made = 0
        while True:
            state = self.get_game_state()
            
            if state["status"] != "playing":
                print(f"\nGame over! Status: {state['status']}")
                break
            
            # Check if we've reached the maximum lines cleared
            if state["lines_cleared"] >= self.max_lines:
                print(f"\nReached maximum lines cleared ({self.max_lines}) - stopping game")
                break
            
            board = state["board"]
            current_piece = state["current_piece"]
            
            if current_piece is None:
                print("No current piece available")
                break
            
            piece_type = PieceType[current_piece["type"]]
            height = len(board)
            width = len(board[0])
            
            # Find best placement using current board state
            # Get fresh valid placements to ensure we're working with current state
            # Make a deep copy of board to ensure we're working with current state
            board_copy = [row[:] for row in board]
            valid_placements = self.get_valid_placements(board_copy, piece_type)
            
            if not valid_placements:
                placement = None
            else:
                # For I-piece: Favor vertical placements (rotation 0 or 2)
                if piece_type == PieceType.I:
                    vertical_placements = []
                    horizontal_placements = []
                    
                    for row, col, rotation in valid_placements:
                        # Check if this is vertical (rotation 0 or 2)
                        if rotation == 0 or rotation == 2:
                            # Vertical I-piece - prefer these
                            # Check column height - allow if column is empty or would stay within 12 blocks
                            test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
                            test_cells = test_piece.get_cells()
                            piece_cols = set(c for _, c in test_cells if 0 <= c < width)
                            piece_col = list(piece_cols)[0] if piece_cols else -1
                            
                            if 0 <= piece_col < width:
                                current_col_height = 0
                                for r in range(height):
                                    if board_copy[r][piece_col] != 0:
                                        current_col_height = height - r
                                        break
                                
                                # Allow if column is empty (height 0) or would stay within 12 blocks after placement
                                if current_col_height == 0 or current_col_height + 4 <= 12:
                                    vertical_placements.append((row, col, rotation))
                        # Check if this is horizontal (rotation 1 or 3)
                        elif rotation == 1 or rotation == 3:
                            # Double-check that this horizontal I-piece can actually rest
                            # Valid placements should already ensure this, but verify with current board
                            test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
                            test_cells = test_piece.get_cells()
                            if test_cells:
                                lowest_row = max(r for r, _ in test_cells if 0 <= r < height)
                                can_rest = (lowest_row == height - 1) or \
                                          any(r == lowest_row and r + 1 < height and board_copy[r + 1][c] != 0 
                                              for r, c in test_cells if 0 <= r < height and 0 <= c < width)
                                
                                # Check if this horizontal placement would cause any column to exceed 12 blocks
                                # BUT: if it clears lines, allow it even if it would temporarily exceed the limit
                                piece_cols = set(c for _, c in test_cells if 0 <= c < width)
                                would_exceed_limit = False
                                max_i_piece_height = 12  # 3 I-pieces * 4 blocks each
                                
                                # Simulate placement to check column heights and line clearing
                                test_board_sim = [r[:] for r in board_copy]
                                for r, c in test_cells:
                                    if 0 <= r < height and 0 <= c < width:
                                        test_board_sim[r][c] = 1
                                
                                # Check if this placement clears lines
                                clears_lines = False
                                cleared_rows = []
                                for r in range(height):
                                    if all(cell != 0 for cell in test_board_sim[r]):
                                        clears_lines = True
                                        cleared_rows.append(r)
                                
                                if clears_lines:
                                    # If it clears lines, check final height after clearing
                                    final_board = []
                                    for r in range(height):
                                        if r not in cleared_rows:
                                            final_board.append(test_board_sim[r])
                                    while len(final_board) < height:
                                        final_board.insert(0, [0] * width)
                                    
                                    max_final_height = 0
                                    for c in range(width):
                                        col_height = 0
                                        for r in range(height):
                                            if final_board[r][c] != 0:
                                                col_height = height - r
                                                break
                                        max_final_height = max(max_final_height, col_height)
                                    
                                    # Allow if final height after clearing is <= 12 blocks  
                                    # Allow a small tolerance (1 block) to account for edge cases
                                    if max_final_height <= 13:  # Allow up to 13 to account for rounding/edge cases
                                        horizontal_placements.append((row, col, rotation))
                                    # If it clears but would still exceed 13, reject it
                                    else:
                                        pass  # Would exceed limit even after clearing
                                else:
                                    # Doesn't clear lines - check if it would exceed limit
                                    for c in piece_cols:
                                        if 0 <= c < width:
                                            col_height_after = 0
                                            for r in range(height):
                                                if test_board_sim[r][c] != 0:
                                                    col_height_after = height - r
                                                    break
                                            if col_height_after > max_i_piece_height:
                                                would_exceed_limit = True
                                                break
                                    
                                    if can_rest and not would_exceed_limit:
                                        horizontal_placements.append((row, col, rotation))
                            else:
                                horizontal_placements.append((row, col, rotation))

                    
                    # For I-piece: default vertical > horizontal. When --preferred-rotation is set on a
                    # single-piece game, prefer that orientation first so rotation coverage can be filled
                    # (find_best_placement also encodes this but play_game uses this path instead).
                    pref = self.preferred_rotation
                    single_piece = len(self.piece_types) == 1
                    want_horizontal = single_piece and pref is not None and pref in (1, 3)
                    want_vertical = single_piece and pref is not None and pref in (0, 2)

                    if want_horizontal:
                        if horizontal_placements:
                            placements_to_evaluate = horizontal_placements
                        elif vertical_placements:
                            placements_to_evaluate = vertical_placements
                        else:
                            placements_to_evaluate = []
                    elif want_vertical:
                        if vertical_placements:
                            placements_to_evaluate = vertical_placements
                        elif horizontal_placements:
                            placements_to_evaluate = horizontal_placements
                        else:
                            placements_to_evaluate = []
                    else:
                        if vertical_placements:
                            placements_to_evaluate = vertical_placements
                        elif horizontal_placements:
                            placements_to_evaluate = horizontal_placements
                        else:
                            placements_to_evaluate = []
                else:
                    placements_to_evaluate = valid_placements
                
                # Evaluate all valid placements and select the best one
                best_placement = None
                best_score = float('-inf')
                
                for row, col, rotation in placements_to_evaluate:
                    # Final validation before evaluating - ensure placement is still valid
                    test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
                    test_cells = test_piece.get_cells()
                    
                    # Check bounds and emptiness (using board_copy for consistency)
                    if not all(0 <= r < height and 0 <= c < width for r, c in test_cells):
                        continue
                    if not all(board_copy[r][c] == 0 for r, c in test_cells if 0 <= r < height and 0 <= c < width):
                        continue
                    
                    # Check if can rest (using board_copy for consistency)
                    if test_cells:
                        lowest_row = max(r for r, _ in test_cells if 0 <= r < height)
                        can_rest = (lowest_row == height - 1) or \
                                  any(r == lowest_row and r + 1 < height and board_copy[r + 1][c] != 0 
                                      for r, c in test_cells if 0 <= r < height and 0 <= c < width)
                        if not can_rest:
                            continue  # Skip floating placements
                    
                    # For I-piece: double-check that this placement is in our filtered list
                    if piece_type == PieceType.I:
                        # Verify this is actually a valid placement we want to consider
                        if (row, col, rotation) not in placements_to_evaluate:
                            continue  # Shouldn't happen, but safety check
                    
                    score = self.evaluate_placement(board_copy, row, col, rotation, piece_type)
                    if (
                        self.preferred_rotation is not None
                        and len(self.piece_types) == 1
                        and rotation == self.preferred_rotation
                    ):
                        score += 2000.0
                    if score > best_score:
                        best_score = score
                        best_placement = (row, col, rotation)
                
                placement = best_placement
                
                # Final safety check: verify placement is in valid list
                if placement is not None and placement not in valid_placements:
                    # This shouldn't happen, but if it does, try to find a valid one
                    print(f"  WARNING: Selected placement {placement} not in valid list, finding alternative...")
                    if placements_to_evaluate:
                        placement = placements_to_evaluate[0]  # Use first from filtered list
                    else:
                        placement = None
            
            if placement is None:
                # Debug: check if there are any valid placements at all
                valid_placements = self.get_valid_placements(board, piece_type)
                print(f"\nNo valid placement found - game over")
                print(f"  Piece type: {piece_type.value}")
                print(f"  Valid placements found: {len(valid_placements)}")
                print(f"  Board dimensions: {len(board)} rows x {len(board[0])} cols")
                print(f"  Board height map: {self._get_height_map(board)}")
                
                # Show a sample of the board
                print(f"  Board sample (bottom 3 rows):")
                for row_idx in range(max(0, len(board)-3), len(board)):
                    row_str = ''.join('█' if cell else '·' for cell in board[row_idx])
                    print(f"    Row {row_idx}: {row_str}")
                
                if len(valid_placements) == 0:
                    # Try to find placements without the floating check
                    print(f"  Attempting to find any valid position...")
                    # Try a simple placement at the bottom
                    for rotation in range(4):
                        test_piece = Piece(piece_type, row=19, col=5, rotation=rotation)
                        cells = test_piece.get_cells()
                        if all(0 <= r < len(board) and 0 <= c < len(board[0]) for r, c in cells):
                            if all(board[r][c] == 0 for r, c in cells):
                                print(f"  Found potential placement at row=19, col=5, rotation={rotation}")
                                print(f"    Cells: {cells}")
                break
            
            row, col, rotation = placement
            
            # Place the piece
            success = self.place_piece(row, col, rotation)
            
            if not success:
                # Get API error message
                try:
                    response = requests.post(
                        f"{self.api_url}/games/{self.game_id}/place",
                        json={"row": row, "col": col, "rotation": rotation}
                    )
                    error_msg = response.json().get("detail", "Unknown error") if response.status_code != 200 else "Success"
                except:
                    error_msg = "Could not get error message"
                
                print(f"Failed to place piece at ({row}, {col}, {rotation})")
                print(f"  API error: {error_msg}")
                
                # Re-check valid placements with current board state
                current_state = self.get_game_state()
                current_board = current_state["board"]
                valid_placements = self.get_valid_placements(current_board, piece_type)
                print(f"  Valid placements available: {len(valid_placements)}")
                
                # Check if this placement is actually in the valid list
                if (row, col, rotation) in valid_placements:
                    print(f"  WARNING: Placement ({row}, {col}, {rotation}) is in valid list but API rejected it!")
                    # Verify the placement manually
                    test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
                    cells = test_piece.get_cells()
                    print(f"  Piece cells: {cells}")
                    print(f"  All in bounds: {all(0 <= r < len(current_board) and 0 <= c < len(current_board[0]) for r, c in cells)}")
                    print(f"  All empty: {all(current_board[r][c] == 0 for r, c in cells if 0 <= r < len(current_board) and 0 <= c < len(current_board[0]))}")
                    # Check if floating
                    lowest_row = max(r for r, _ in cells) if cells else -1
                    can_rest = (lowest_row == len(current_board) - 1) or \
                              any(r == lowest_row and r + 1 < len(current_board) and current_board[r + 1][c] != 0 
                                  for r, c in cells)
                    print(f"  Can rest: {can_rest} (lowest row: {lowest_row})")
                else:
                    print(f"  Placement ({row}, {col}, {rotation}) is NOT in valid placements list")
                    if valid_placements:
                        print(f"  First few valid placements: {valid_placements[:5]}")
                break
            
            moves_made += 1
            new_state = self.get_game_state()
            
            if moves_made % 10 == 0:
                print(f"  Moves: {moves_made}, Score: {new_state['score']}, "
                      f"Lines: {new_state['lines_cleared']}/{self.max_lines}")
        
        # Get final state
        final_state = self.get_game_state()
        print(f"\nFinal Score: {final_state['score']}")
        print(f"Lines Cleared: {final_state['lines_cleared']}")
        print(f"Pieces Placed: {final_state['pieces_placed']}")
        
        # Download game data
        response = requests.get(f"{self.api_url}/games/{self.game_id}/moves")
        response.raise_for_status()
        game_data = response.json()
        
        return game_data
    
    def save_game_data(self, game_data: Dict, output_dir: str = ".") -> str:
        """Save game data to a file.
        
        Args:
            game_data: Game data dictionary
            output_dir: Directory to save the file
            
        Returns:
            Path to saved file
        """
        lines_cleared = game_data["final_lines_cleared"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Tag with placement direction so analyzers can classify the file correctly
        game_data["placement_direction"] = self.placement_direction
        # Create filename from piece types (e.g., "I" or "I_L" or "I_O_T_S_Z_J_L")
        piece_str = "_".join(self.piece_types)
        filename = f"{piece_str}_{lines_cleared}lines_{timestamp}.json"
        filepath = f"{output_dir}/{filename}"
        
        with open(filepath, "w") as f:
            json.dump(game_data, f, indent=2)
        
        print(f"Saved game data to: {filepath}")
        return filepath


def main():
    """Main function to run the heuristic data generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate puzzle grid training data using heuristics"
    )
    parser.add_argument(
        "--pieces",
        type=str,
        nargs="+",
        required=True,
        choices=[p.name for p in PieceType],
        help="Piece type(s) to use (1 to 7 pieces: e.g., 'I' or 'I L' or 'I O T S Z J L')"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save game data files (default: current directory)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=BASE_URL,
        help=f"Base URL of the PuzzleGrid API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1000,
        help="Maximum number of lines to clear before stopping (default: 1000)"
    )
    parser.add_argument(
        "--placement-direction",
        type=str,
        default="left-to-right",
        choices=["left-to-right", "right-to-left"],
        help="Direction to favor when placing pieces: 'left-to-right' (default) or 'right-to-left'"
    )
    parser.add_argument(
        "--preferred-rotation",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Preferred rotation (0-3) for single-piece games. When multiple valid placements exist, prefer this rotation. Only used for single-piece games."
    )
    
    args = parser.parse_args()
    
    # Validate piece count
    if len(args.pieces) == 0:
        parser.error("At least one piece type must be provided")
    if len(args.pieces) > 7:
        parser.error("Maximum of 7 piece types allowed (I, O, T, S, Z, J, L)")
    
    print("=" * 60)
    print("PuzzleGrid Heuristic Data Generator")
    print("=" * 60)
    piece_types_str = ", ".join(args.pieces)
    print(f"Piece Types: {piece_types_str}")
    print(f"Number of Games: {args.games}")
    print(f"Maximum Lines: {args.max_lines}")
    print(f"Placement Direction: {args.placement_direction}")
    if args.preferred_rotation is not None:
        print(f"Preferred Rotation: {args.preferred_rotation}")
    print(f"Output Directory: {args.output_dir}")
    print(f"API URL: {args.api_url}")
    print("=" * 60)
    
    # Check if API is available
    try:
        response = requests.get(f"{args.api_url}/")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to API at {args.api_url}")
        print("Make sure the API server is running:")
        print("  uv run python main.py")
        return
    
    # Play games
    for game_num in range(1, args.games + 1):
        print(f"\n{'=' * 60}")
        print(f"Game {game_num}/{args.games}")
        print(f"{'=' * 60}")
        
        player = HeuristicPlayer(
            args.pieces,
            api_url=args.api_url,
            max_lines=args.max_lines,
            placement_direction=args.placement_direction,
            preferred_rotation=args.preferred_rotation
        )
        game_data = player.play_game()
        player.save_game_data(game_data, output_dir=args.output_dir)
        
        # Small delay between games
        if game_num < args.games:
            time.sleep(0.5)
    
    print(f"\n{'=' * 60}")
    print(f"Completed {args.games} game(s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

