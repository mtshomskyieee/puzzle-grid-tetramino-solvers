"""Puzzle grid pieces (polyominoes) definitions."""

from enum import Enum
from typing import List, Tuple


class PieceType(Enum):
    """Types of puzzle grid pieces."""
    I = "I"
    O = "O"
    T = "T"
    S = "S"
    Z = "Z"
    J = "J"
    L = "L"


class Piece:
    """Represents a puzzle grid piece with rotation states."""
    
    # Piece shapes in their initial rotation (0 degrees)
    # Each piece is defined as a list of (row, col) offsets from the center
    PIECE_SHAPES = {
        PieceType.I: [
            [(-1, 0), (0, 0), (1, 0), (2, 0)],  # 0°
            [(0, -1), (0, 0), (0, 1), (0, 2)],  # 90°
            [(-1, 0), (0, 0), (1, 0), (2, 0)],  # 180° (same as 0°)
            [(0, -1), (0, 0), (0, 1), (0, 2)],  # 270° (same as 90°)
        ],
        PieceType.O: [
            [(0, 0), (0, 1), (1, 0), (1, 1)],  # All rotations are the same
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        ],
        PieceType.T: [
            [(-1, 0), (0, -1), (0, 0), (0, 1)],  # 0°
            [(-1, 0), (0, 0), (0, 1), (1, 0)],   # 90°
            [(0, -1), (0, 0), (0, 1), (1, 0)],   # 180°
            [(-1, 0), (0, -1), (0, 0), (1, 0)],  # 270°
        ],
        PieceType.S: [
            [(-1, 1), (0, 0), (0, 1), (1, 0)],   # 0°
            [(-1, 0), (0, 0), (0, 1), (1, 1)],   # 90°
            [(-1, 1), (0, 0), (0, 1), (1, 0)],   # 180° (same as 0°)
            [(-1, 0), (0, 0), (0, 1), (1, 1)],   # 270° (same as 90°)
        ],
        PieceType.Z: [
            [(-1, 0), (0, 0), (0, 1), (1, 1)],   # 0°
            [(-1, 1), (0, 0), (0, 1), (1, 0)],   # 90°
            [(-1, 0), (0, 0), (0, 1), (1, 1)],   # 180° (same as 0°)
            [(-1, 1), (0, 0), (0, 1), (1, 0)],   # 270° (same as 90°)
        ],
        PieceType.J: [
            [(-1, -1), (-1, 0), (0, 0), (1, 0)], # 0°
            [(-1, 1), (0, 1), (0, 0), (0, -1)],  # 90°
            [(1, 1), (1, 0), (0, 0), (-1, 0)],   # 180°
            [(1, -1), (0, -1), (0, 0), (0, 1)],  # 270°
        ],
        PieceType.L: [
            [(-1, 1), (-1, 0), (0, 0), (1, 0)],  # 0°
            [(-1, -1), (0, -1), (0, 0), (0, 1)], # 90°
            [(1, -1), (1, 0), (0, 0), (-1, 0)],  # 180°
            [(1, 1), (0, 1), (0, 0), (0, -1)],   # 270°
        ],
    }
    
    def __init__(self, piece_type: PieceType, row: int = 0, col: int = 4, rotation: int = 0):
        """Initialize a piece.
        
        Args:
            piece_type: Type of the piece
            row: Starting row position
            col: Starting column position
            rotation: Rotation state (0-3)
        """
        self.piece_type = piece_type
        self.row = row
        self.col = col
        self.rotation = rotation % 4
    
    def get_cells(self) -> List[Tuple[int, int]]:
        """Get the absolute positions of all cells occupied by this piece.
        
        Returns:
            List of (row, col) tuples
        """
        shape = self.PIECE_SHAPES[self.piece_type][self.rotation]
        return [(self.row + dr, self.col + dc) for dr, dc in shape]
    
    def rotate(self) -> 'Piece':
        """Create a new piece rotated clockwise."""
        return Piece(self.piece_type, self.row, self.col, (self.rotation + 1) % 4)
    
    def move(self, dr: int, dc: int) -> 'Piece':
        """Create a new piece moved by the given offsets.
        
        Args:
            dr: Row offset
            dc: Column offset
            
        Returns:
            New piece with updated position
        """
        return Piece(self.piece_type, self.row + dr, self.col + dc, self.rotation)
    
    def copy(self) -> 'Piece':
        """Create a copy of this piece."""
        return Piece(self.piece_type, self.row, self.col, self.rotation)
    
    def __repr__(self) -> str:
        return f"Piece({self.piece_type.value}, row={self.row}, col={self.col}, rot={self.rotation})"

