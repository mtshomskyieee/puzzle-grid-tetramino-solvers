"""Puzzle grid board logic."""

from typing import List, Optional, Set, Tuple


class Board:
    """Represents the puzzle grid game board."""
    
    def __init__(self, width: int = 10, height: int = 20):
        """Initialize an empty board.
        
        Args:
            width: Board width (default 10)
            height: Board height (default 20)
        """
        self.width = width
        self.height = height
        self.grid: List[List[int]] = [[0 for _ in range(width)] for _ in range(height)]
    
    def is_valid_position(self, cells: List[Tuple[int, int]]) -> bool:
        """Check if the given cells are valid positions on the board.
        
        Args:
            cells: List of (row, col) positions
            
        Returns:
            True if all cells are within bounds and empty
        """
        for row, col in cells:
            # Check bounds
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                return False
            # Check if cell is occupied
            if self.grid[row][col] != 0:
                return False
        return True
    
    def can_piece_rest(self, cells: List[Tuple[int, int]]) -> bool:
        """Check if a piece can rest at the given position (not floating).
        
        A piece can rest if:
        - It's at the bottom of the board, OR
        - At least one cell in the piece has a filled cell directly below it
        
        Args:
            cells: List of (row, col) positions for the piece
            
        Returns:
            True if the piece can rest at this position
        """
        if not cells:
            return False
        
        # Find the lowest row in the piece
        lowest_row = max(row for row, _ in cells if 0 <= row < self.height)
        
        # If at the bottom, it can rest
        if lowest_row == self.height - 1:
            return True
        
        # Check if any cell in the piece has support below
        for row, col in cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                # Check if there's a filled cell directly below
                if row + 1 < self.height and self.grid[row + 1][col] != 0:
                    return True
        
        return False
    
    def place_piece(self, cells: List[Tuple[int, int]], piece_id: int) -> None:
        """Place a piece on the board.
        
        Args:
            cells: List of (row, col) positions to fill
            piece_id: ID to mark these cells with (typically piece type enum value)
        """
        for row, col in cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                self.grid[row][col] = piece_id
    
    def clear_lines(self) -> int:
        """Clear all full lines and return the number of lines cleared.
        
        Returns:
            Number of lines cleared
        """
        lines_cleared = 0
        new_grid = []
        
        for row in self.grid:
            if all(cell != 0 for cell in row):
                lines_cleared += 1
            else:
                new_grid.append(row)
        
        # Add empty rows at the top
        while len(new_grid) < self.height:
            new_grid.insert(0, [0] * self.width)
        
        self.grid = new_grid
        return lines_cleared
    
    def get_state(self) -> List[List[int]]:
        """Get a copy of the current board state.
        
        Returns:
            2D list representing the board
        """
        return [row[:] for row in self.grid]
    
    def get_height_map(self) -> List[int]:
        """Get the height of each column.
        
        Returns:
            List of heights, one per column
        """
        heights = []
        for col in range(self.width):
            height = 0
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    height = self.height - row
                    break
            heights.append(height)
        return heights
    
    def has_floating_pieces(self) -> bool:
        """Check if there are any floating pieces on the board.
        
        A piece is floating if it has empty space below it and is not at the bottom.
        
        Returns:
            True if there are floating pieces, False otherwise
        """
        for row in range(self.height - 1):  # Don't check bottom row
            for col in range(self.width):
                if self.grid[row][col] != 0:
                    # Check if there's empty space below this cell
                    if self.grid[row + 1][col] == 0:
                        # This cell has a piece but empty space below - check if it's floating
                        # A piece is floating if all cells in its "group" have empty space below
                        # For simplicity, we'll check if this cell and adjacent cells of the same piece
                        # have support below
                        piece_id = self.grid[row][col]
                        has_support = False
                        
                        # Check if any cell in the same column below this one is filled
                        for check_row in range(row + 1, self.height):
                            if self.grid[check_row][col] != 0:
                                has_support = True
                                break
                        
                        # Also check adjacent columns for support (pieces can rest on adjacent pieces)
                        if not has_support and col > 0:
                            if self.grid[row + 1][col - 1] != 0:
                                has_support = True
                        if not has_support and col < self.width - 1:
                            if self.grid[row + 1][col + 1] != 0:
                                has_support = True
                        
                        if not has_support:
                            return True
        return False
    
    def __repr__(self) -> str:
        """String representation of the board."""
        lines = []
        for row in self.grid:
            line = "".join("█" if cell else "·" for cell in row)
            lines.append(f"|{line}|")
        return "\n".join(lines)

