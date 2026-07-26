"""Puzzle grid game state and logic."""

import random
from typing import List, Optional, Set
from enum import Enum

from .board import Board
from .pieces import Piece, PieceType


class GameStatus(Enum):
    """Game status enumeration."""
    PLAYING = "playing"
    GAME_OVER = "game_over"
    PAUSED = "paused"


class MoveType(Enum):
    """Types of moves that can be made."""
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    ROTATE = "rotate"
    DROP = "drop"
    PLACE = "place"  # Direct placement for RL


class GameMove:
    """Represents a single game move for logging."""
    
    def __init__(
        self,
        move_type: MoveType,
        piece_type: Optional[PieceType] = None,
        piece_position: Optional[dict] = None,
        lines_cleared: int = 0,
        board_state: Optional[List[List[int]]] = None,
    ):
        self.move_type = move_type
        self.piece_type = piece_type
        self.piece_position = piece_position  # {row, col, rotation}
        self.lines_cleared = lines_cleared
        self.board_state = board_state
    
    def to_dict(self) -> dict:
        """Convert move to dictionary for JSON serialization."""
        return {
            "move_type": self.move_type.value,
            "piece_type": self.piece_type.value if self.piece_type else None,
            "piece_position": self.piece_position,
            "lines_cleared": self.lines_cleared,
            "board_state": self.board_state,
        }


class PuzzleGridGame:
    """Main puzzle grid game class."""
    
    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        allowed_pieces: Optional[List[PieceType]] = None,
        game_id: Optional[str] = None,
        piece_sequence: Optional[List[str]] = None,
    ):
        """Initialize a new puzzle grid game.
        
        Args:
            width: Board width
            height: Board height
            allowed_pieces: List of allowed piece types. If None, all pieces are allowed.
            game_id: Optional game identifier
            piece_sequence: Optional list of piece type names (e.g., ["I", "J", "L"]) for deterministic spawning.
                          If provided, pieces will be spawned from this sequence instead of randomly.
        """
        self.game_id = game_id or f"game_{random.randint(1000, 9999)}"
        self.board = Board(width, height)
        self.status = GameStatus.PLAYING
        
        # Piece configuration
        if allowed_pieces is None:
            self.allowed_pieces = list(PieceType)
        else:
            self.allowed_pieces = allowed_pieces.copy()
        
        # Piece sequence for deterministic spawning
        self.piece_sequence: Optional[List[PieceType]] = None
        self.piece_sequence_index: int = 0
        if piece_sequence:
            # Convert string names to PieceType enums
            self.piece_sequence = [PieceType[name.upper()] for name in piece_sequence]
        
        # Track spawned pieces for recording
        self.spawned_pieces: List[str] = []
        
        # Current piece
        self.current_piece: Optional[Piece] = None
        self.next_piece: Optional[Piece] = None
        
        # Game statistics
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        
        # Last placed piece info for debugging
        self.last_placed_piece: Optional[dict] = None
        
        # Move history for replay and training
        self.move_history: List[GameMove] = []
        
        # Spawn the first piece
        self._spawn_next_piece()
    
    def _get_next_piece_type(self) -> PieceType:
        """Get the next piece type from sequence or random.
        
        Returns:
            Next piece type to spawn
        """
        if self.piece_sequence and self.piece_sequence_index < len(self.piece_sequence):
            # Use piece from sequence
            piece_type = self.piece_sequence[self.piece_sequence_index]
            self.piece_sequence_index += 1
            # Record spawned piece
            self.spawned_pieces.append(piece_type.name)
        else:
            # Fall back to random (shouldn't happen in normal play with sequence)
            piece_type = random.choice(self.allowed_pieces)
            # Record spawned piece
            self.spawned_pieces.append(piece_type.name)
        
        return piece_type
    
    def _spawn_next_piece(self) -> bool:
        """Spawn the next piece. Returns False if game over."""
        if self.next_piece is None:
            piece_type = self._get_next_piece_type()
            # Calculate spawn row to account for pieces that extend above row 0
            spawn_row = self._calculate_spawn_row(piece_type)
            self.next_piece = Piece(piece_type, row=spawn_row, col=self.board.width // 2 - 1)
        
        self.current_piece = self.next_piece
        
        # Check if we can place the piece
        if not self.board.is_valid_position(self.current_piece.get_cells()):
            self.status = GameStatus.GAME_OVER
            return False
        
        # Generate next piece
        piece_type = self._get_next_piece_type()
        spawn_row = self._calculate_spawn_row(piece_type)
        self.next_piece = Piece(piece_type, row=spawn_row, col=self.board.width // 2 - 1)
        
        return True
    
    def _calculate_spawn_row(self, piece_type: PieceType) -> int:
        """Calculate the spawn row for a piece type to ensure it fits on the board.
        
        Args:
            piece_type: Type of piece to spawn
            
        Returns:
            Row position where the piece should spawn
        """
        # Get the shape for rotation 0 (initial spawn rotation)
        shape = Piece.PIECE_SHAPES[piece_type][0]
        # Find the minimum row offset (most negative)
        min_row_offset = min(dr for dr, _ in shape)
        # Spawn at row 0 if min_offset is 0 or positive, otherwise spawn higher
        # to account for negative offsets
        if min_row_offset < 0:
            return -min_row_offset  # Spawn higher to accommodate negative offsets
        return 0
    
    def _lock_piece(self) -> int:
        """Lock the current piece on the board and clear lines.
        
        Returns:
            Number of lines cleared
        """
        if self.current_piece is None:
            return 0
        
        # Place piece on board (using piece type enum value as ID)
        piece_id = list(PieceType).index(self.current_piece.piece_type) + 1
        self.board.place_piece(self.current_piece.get_cells(), piece_id)
        
        # Clear lines
        lines_cleared = self.board.clear_lines()
        
        # Update statistics
        self.lines_cleared_total += lines_cleared
        self.pieces_placed += 1
        
        # Scoring: 100 * (lines_cleared ^ 2)
        if lines_cleared > 0:
            self.score += 100 * (lines_cleared ** 2)
        
        self.current_piece = None
        return lines_cleared
    
    def move(self, move_type: MoveType) -> bool:
        """Make a move in the game.
        
        Args:
            move_type: Type of move to make
            
        Returns:
            True if move was successful, False otherwise
        """
        if self.status != GameStatus.PLAYING or self.current_piece is None:
            return False
        
        new_piece = None
        
        if move_type == MoveType.LEFT:
            new_piece = self.current_piece.move(0, -1)
        elif move_type == MoveType.RIGHT:
            new_piece = self.current_piece.move(0, 1)
        elif move_type == MoveType.DOWN:
            new_piece = self.current_piece.move(1, 0)
        elif move_type == MoveType.ROTATE:
            new_piece = self.current_piece.rotate()
        elif move_type == MoveType.DROP:
            # Drop piece to lowest valid position
            new_piece = self.current_piece.copy()
            while self.board.is_valid_position(new_piece.move(1, 0).get_cells()):
                new_piece = new_piece.move(1, 0)
        else:
            return False
        
        # Validate move
        if self.board.is_valid_position(new_piece.get_cells()):
            self.current_piece = new_piece
            
            # If moving down, check if we need to lock
            if move_type == MoveType.DOWN or move_type == MoveType.DROP:
                # Check if piece can move down further
                if not self.board.is_valid_position(self.current_piece.move(1, 0).get_cells()):
                    lines_cleared = self._lock_piece()
                    self._log_move(move_type, lines_cleared)
                    self._spawn_next_piece()
                    return True
            
            self._log_move(move_type, 0)
            return True
        
        return False
    
    def _drop_piece_to_rest(self, piece: Piece) -> Piece:
        """Drop a piece down to its lowest valid resting position.
        
        Args:
            piece: Piece to drop
            
        Returns:
            Piece at its lowest valid resting position
        """
        # Start from the piece's current position and drop it down
        current_piece = piece
        dropped_piece = piece
        
        # Keep moving down until we can't move further
        while True:
            # Try moving down one row
            test_piece = current_piece.move(1, 0)
            
            # Check if the new position is valid
            if self.board.is_valid_position(test_piece.get_cells()):
                dropped_piece = test_piece
                current_piece = test_piece
            else:
                # Can't move down further, this is the resting position
                break
        
        return dropped_piece
    
    def place_piece_direct(
        self,
        row: int,
        col: int,
        rotation: int = 0,
        drop_to_rest: bool = True,
    ) -> bool:
        """Directly place a piece at a specific position (for RL training).
        
        Args:
            row: Row position
            col: Column position
            rotation: Rotation state (0-3)
            drop_to_rest: If True, drop the piece to its resting position before locking
            
        Returns:
            True if placement was successful
        """
        if self.status != GameStatus.PLAYING or self.current_piece is None:
            return False
        
        # Create piece at specified position
        new_piece = Piece(
            self.current_piece.piece_type,
            row=row,
            col=col,
            rotation=rotation % 4
        )
        
        # Validate initial placement
        if not self.board.is_valid_position(new_piece.get_cells()):
            return False
        
        # If drop_to_rest is True, drop the piece to its resting position
        if drop_to_rest:
            new_piece = self._drop_piece_to_rest(new_piece)
            # Re-validate after dropping (should still be valid)
            if not self.board.is_valid_position(new_piece.get_cells()):
                return False
        
        # Store last placed piece info before locking (use final position after dropping)
        piece_type_value = new_piece.piece_type.value
        piece_type_enum = new_piece.piece_type  # Keep enum for logging
        final_row = new_piece.row
        final_col = new_piece.col
        final_rotation = new_piece.rotation
        
        self.last_placed_piece = {
            "type": piece_type_value,
            "row": final_row,
            "col": final_col,
            "rotation": final_rotation,
        }
        
        # Place the piece
        self.current_piece = new_piece
        
        # Lock the piece (this sets current_piece to None)
        lines_cleared = self._lock_piece()
        
        # Log move with piece_type captured before locking
        self._log_move(
            MoveType.PLACE,
            lines_cleared,
            piece_position={"row": final_row, "col": final_col, "rotation": final_rotation},
            piece_type=piece_type_enum  # Pass piece_type explicitly
        )
        
        # Spawn next piece
        if not self._spawn_next_piece():
            return False
        
        return True
    
    def _log_move(self, move_type: MoveType, lines_cleared: int, piece_position: Optional[dict] = None, piece_type: Optional[PieceType] = None):
        """Log a move to history.
        
        Args:
            move_type: Type of move
            lines_cleared: Number of lines cleared
            piece_position: Optional piece position dict (if None, uses current_piece)
            piece_type: Optional piece type (if None, uses current_piece.piece_type)
        """
        # Use provided piece_type, or fall back to current_piece
        logged_piece_type = piece_type if piece_type is not None else (self.current_piece.piece_type if self.current_piece else None)
        
        move = GameMove(
            move_type=move_type,
            piece_type=logged_piece_type,
            piece_position=piece_position or (
                {
                    "row": self.current_piece.row,
                    "col": self.current_piece.col,
                    "rotation": self.current_piece.rotation
                } if self.current_piece else None
            ),
            lines_cleared=lines_cleared,
            board_state=self.board.get_state(),
        )
        self.move_history.append(move)
    
    def get_state(self) -> dict:
        """Get the current game state.
        
        Returns:
            Dictionary containing game state
        """
        current_piece_pos = None
        if self.current_piece:
            current_piece_pos = {
                "type": self.current_piece.piece_type.value,
                "row": self.current_piece.row,
                "col": self.current_piece.col,
                "rotation": self.current_piece.rotation,
                "cells": self.current_piece.get_cells(),
            }
        
        next_piece_type = None
        if self.next_piece:
            next_piece_type = self.next_piece.piece_type.value
        
        return {
            "game_id": self.game_id,
            "status": self.status.value,
            "board": self.board.get_state(),
            "current_piece": current_piece_pos,
            "next_piece": next_piece_type,
            "last_placed_piece": self.last_placed_piece,
            "has_floating_pieces": self.board.has_floating_pieces(),
            "score": self.score,
            "lines_cleared": self.lines_cleared_total,
            "pieces_placed": self.pieces_placed,
            "allowed_pieces": [p.value for p in self.allowed_pieces],
        }
    
    def get_moves_json(self) -> dict:
        """Get move history as JSON-serializable dictionary.
        
        Returns:
            Dictionary containing all moves and game metadata
        """
        result = {
            "game_id": self.game_id,
            "width": self.board.width,
            "height": self.board.height,
            "allowed_pieces": [p.value for p in self.allowed_pieces],
            "final_score": self.score,
            "final_lines_cleared": self.lines_cleared_total,
            "final_pieces_placed": self.pieces_placed,
            "moves": [move.to_dict() for move in self.move_history],
        }
        
        # Include piece sequence if it was recorded
        if self.spawned_pieces:
            result["piece_sequence"] = self.spawned_pieces
        
        return result

