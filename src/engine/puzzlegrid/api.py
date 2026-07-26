"""FastAPI application for puzzle grid game."""

import json
import os
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Set
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from .game import PuzzleGridGame, GameStatus, MoveType
from .pieces import Piece, PieceType


app = FastAPI(title="PuzzleGrid API", version="0.1.0", description="Puzzle grid game API for reinforcement learning")

# In-memory game storage (in production, use a database)
games: dict[str, PuzzleGridGame] = {}

# Replay state storage
replay_games: dict[str, dict] = {}  # game_id -> {training_data, current_move_index, game_instance}

# WebSocket connections for each game
websocket_connections: dict[str, Set[WebSocket]] = {}  # game_id -> set of websockets

# Background tasks for replay advancement
replay_tasks: dict[str, asyncio.Task] = {}


def get_potential_placements(game: PuzzleGridGame) -> dict:
    """Return the exact legal placements currently offered to the active piece."""
    if game.current_piece is None:
        return {
            "piece_type": None,
            "placements": [],
            "count": 0,
        }

    from puzzlegrid import placement

    piece_type = game.current_piece.piece_type
    board_state = game.board.get_state()
    valid = placement.get_valid_placements(board_state, piece_type)
    valid = placement.filter_low_stack(board_state, piece_type, valid)
    placements = []
    for index, (row, col, rotation) in enumerate(valid, start=1):
        piece = Piece(piece_type, row=row, col=col, rotation=rotation)
        placements.append(
            {
                "index": index,
                "row": row,
                "col": col,
                "rotation": rotation,
                "cells": piece.get_cells(),
            }
        )

    return {
        "piece_type": piece_type.value,
        "placements": placements,
        "count": len(placements),
    }


def get_training_data_dir():
    """Get the absolute path to the training data directory.

    Layout: <repo_root>/src/engine/puzzlegrid/api.py -> repo_root is 4 levels up.
    Data ships in <repo_root>/data/training_data_clean.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return os.path.join(repo_root, "data", "training_data_clean")


async def replay_advancement_task(game_id: str):
    """Background task that advances replay moves every 0.1 seconds.
    
    Args:
        game_id: Replay game ID
    """
    while game_id in replay_games:
        replay = replay_games[game_id]
        training_data = replay["training_data"]
        current_index = replay["current_move_index"]
        
        # Stop if replay is complete
        if current_index >= len(training_data["moves"]):
            break
        
        # Advance one move
        try:
            await replay_next_move(game_id)
        except Exception as e:
            print(f"Error advancing replay {game_id}: {e}")
            break
        
        # Wait 0.1 seconds before next move (faster replay)
        await asyncio.sleep(0.1)
    
    # Clean up task
    if game_id in replay_tasks:
        del replay_tasks[game_id]


class CreateGameRequest(BaseModel):
    """Request model for creating a new game."""
    width: int = 10
    height: int = 20
    allowed_pieces: Optional[List[str]] = None  # List of piece type strings, None = all pieces
    game_id: Optional[str] = None
    piece_sequence: Optional[List[str]] = None  # List of piece type names for deterministic spawning


class MoveRequest(BaseModel):
    """Request model for making a move."""
    move: str  # "left", "right", "down", "rotate", "drop"


class PlacePieceRequest(BaseModel):
    """Request model for direct piece placement."""
    row: int
    col: int
    rotation: int = 0
    drop_to_rest: bool = True  # If True, drop piece to resting position; if False, place exactly at position


@app.post("/games", response_model=dict)
async def create_game(request: CreateGameRequest):
    """Create a new puzzle grid game.
    
    Args:
        request: Game configuration
        
    Returns:
        Game state
    """
    # Convert piece type strings to PieceType enums
    allowed_pieces = None
    if request.allowed_pieces is not None:
        try:
            allowed_pieces = [PieceType[p.upper()] for p in request.allowed_pieces]
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid piece type: {e}. Valid types: {[p.name for p in PieceType]}"
            )
    
    game = PuzzleGridGame(
        width=request.width,
        height=request.height,
        allowed_pieces=allowed_pieces,
        game_id=request.game_id,
        piece_sequence=request.piece_sequence,
    )
    
    games[game.game_id] = game
    return game.get_state()


@app.get("/games/{game_id}", response_model=dict)
async def get_game_state(game_id: str):
    """Get the current state of a game.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Game state
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return games[game_id].get_state()


@app.post("/games/{game_id}/move", response_model=dict)
async def make_move(game_id: str, request: MoveRequest):
    """Make a move in the game.
    
    Args:
        game_id: Game identifier
        request: Move request
        
    Returns:
        Updated game state
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    
    # Convert string to MoveType
    move_mapping = {mt.value: mt for mt in MoveType}
    move_type = move_mapping.get(request.move.lower())
    if move_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid move: {request.move}. Valid moves: {[m.value for m in MoveType]}"
        )
    
    success = game.move(move_type)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid move or game is not in playing state")
    
    state = game.get_state()
    
    # Broadcast update to WebSocket clients
    result = {"type": "game_update", "game_state": state}
    if game_id in replay_games:
        replay = replay_games[game_id]
        training_data = replay["training_data"]
        current_index = replay["current_move_index"]
        result["replay_info"] = {
            "move_index": current_index,
            "total_moves": len(training_data["moves"]),
            "is_complete": current_index >= len(training_data["moves"]),
            "filename": replay["filename"],
        }
    await broadcast_game_update(game_id, result)
    
    return state


@app.post("/games/{game_id}/place", response_model=dict)
async def place_piece(game_id: str, request: PlacePieceRequest):
    """Directly place the current piece at a specific position (for RL training).
    
    Args:
        game_id: Game identifier
        request: Placement request with row, col, and rotation
        
    Returns:
        Updated game state
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    
    success = game.place_piece_direct(
        row=request.row,
        col=request.col,
        rotation=request.rotation,
        drop_to_rest=request.drop_to_rest,
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid placement or game is not in playing state"
        )
    
    state = game.get_state()
    
    # Broadcast update to WebSocket clients
    result = {"type": "game_update", "game_state": state}
    if game_id in replay_games:
        replay = replay_games[game_id]
        training_data = replay["training_data"]
        current_index = replay["current_move_index"]
        result["replay_info"] = {
            "move_index": current_index,
            "total_moves": len(training_data["moves"]),
            "is_complete": current_index >= len(training_data["moves"]),
            "filename": replay["filename"],
        }
    await broadcast_game_update(game_id, result)
    
    return state


@app.get("/games/{game_id}/moves", response_model=dict)
async def get_moves(game_id: str):
    """Get all moves made in the game as JSON.
    
    Args:
        game_id: Game identifier
        
    Returns:
        JSON object containing all moves and game metadata
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return games[game_id].get_moves_json()


@app.get("/games/{game_id}/potential-placements", response_model=dict)
async def get_game_potential_placements(game_id: str):
    """Get the legal resting placements for the current piece."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]
    placements = get_potential_placements(game)
    return {
        "game_id": game_id,
        **placements,
    }


@app.get("/games/{game_id}/download")
async def download_moves(game_id: str):
    """Download game moves as a JSON file.
    
    Args:
        game_id: Game identifier
        
    Returns:
        JSON file download
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    moves_data = games[game_id].get_moves_json()
    
    return JSONResponse(
        content=moves_data,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="puzzlegrid_game_{game_id}.json"'
        }
    )


@app.delete("/games/{game_id}")
async def delete_game(game_id: str):
    """Delete a game.
    
    Args:
        game_id: Game identifier
        
    Returns:
        Success message
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del games[game_id]
    return {"message": f"Game {game_id} deleted"}


@app.get("/games")
async def list_games():
    """List all active games.
    
    Returns:
        List of game IDs and their statuses
    """
    return {
        "games": [
            {
                "game_id": game_id,
                "status": game.status.value,
                "score": game.score,
                "lines_cleared": game.lines_cleared_total,
            }
            for game_id, game in games.items()
        ]
    }


@app.get("/view", response_class=HTMLResponse)
async def view_games_list():
    """View a list of all games with links to view each one, plus training data replay.
    
    Returns:
        HTML page with list of games and training data selector
    """
    # Get training data files - use absolute path based on project root
    training_dir = get_training_data_dir()
    training_files = []
    if os.path.exists(training_dir):
        training_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
        training_files.sort(reverse=True)  # Most recent first
    
    if not games and not training_files:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PuzzleGrid Games</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }
                .container {
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎮 PuzzleGrid Games</h1>
                <p>No games available. Create a game using the API first.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PuzzleGrid Games</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .section {
                margin-bottom: 40px;
            }
            .section-title {
                font-size: 20px;
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #4a90e2;
            }
            .game-list {
                display: grid;
                gap: 15px;
            }
            .game-item {
                background: #f5f5f5;
                padding: 20px;
                border-radius: 8px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .game-item:hover {
                background: #e8e8e8;
            }
            .game-link {
                text-decoration: none;
                color: #4a90e2;
                font-weight: bold;
                font-size: 18px;
            }
            .game-link:hover {
                color: #357abd;
            }
            .game-stats {
                display: flex;
                gap: 20px;
                font-size: 14px;
                color: #666;
            }
            .stat {
                display: flex;
                flex-direction: column;
            }
            .stat-label {
                font-size: 11px;
                text-transform: uppercase;
            }
            .stat-value {
                font-weight: bold;
                color: #333;
            }
            .status-badge {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .status-playing { background: #2ecc71; color: white; }
            .status-game-over { background: #e74c3c; color: white; }
            .status-paused { background: #f39c12; color: white; }
            .file-list {
                background: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                max-height: 500px;
                overflow-y: auto;
            }
            .file-list-item {
                padding: 12px 15px;
                margin: 6px 0;
                background: white;
                border-radius: 6px;
                border-left: 4px solid #4a90e2;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .file-list-item:hover {
                background: #e8f4f8;
                transform: translateX(3px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.15);
            }
            .file-list-item:active {
                transform: translateX(1px);
            }
            .play-button {
                background: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
                cursor: pointer;
                transition: background 0.2s;
                margin-left: 10px;
            }
            .play-button:hover {
                background: #357abd;
            }
            .file-name {
                flex: 1;
            }
            .file-count {
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
            }
        </style>
        <script>
            async function startReplay(filename) {
                if (!filename) {
                    alert('Please select a training data file');
                    return;
                }
                
                try {
                    const response = await fetch('/replay/start?filename=' + encodeURIComponent(filename), {
                        method: 'POST'
                    });
                    const data = await response.json();
                    if (response.ok) {
                        window.location.href = '/games/' + data.game_id + '/view?replay=true';
                    } else {
                        alert('Error starting replay: ' + (data.detail || 'Unknown error'));
                    }
                } catch (error) {
                    alert('Error starting replay: ' + error.message);
                }
            }
            
            // Handle clicks on file items using data attributes
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('.file-list-item').forEach(function(item) {
                    item.addEventListener('click', function(e) {
                        if (e.target.classList.contains('play-button')) {
                            e.stopPropagation();
                        }
                        const filename = this.getAttribute('data-filename');
                        if (filename) {
                            startReplay(filename);
                        }
                    });
                });
            });
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎮 PuzzleGrid Games</h1>
            
            <div class="section">
                <div class="section-title">📼 Replay Training Data</div>
                <div class="file-list">
                    <div class="file-count">Click a file to start replay ("""
    
    html += f"{len(training_files)}"
    
    html += """ total):</div>
    """
    
    if training_files:
        for filename in training_files:
            # Escape filename for HTML attribute (data-filename)
            html_escaped_filename = filename.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            # Escape filename for HTML display
            display_filename = html_escaped_filename
            html += f'<div class="file-list-item" data-filename="{html_escaped_filename}">'
            html += f'<span class="file-name">{display_filename}</span>'
            html += '<button class="play-button">▶ Play</button>'
            html += '</div>'
    else:
        html += '<p style="text-align: center; color: #666; padding: 20px;">No training data files found in ./data/training_data_clean</p>'
    
    html += """
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🎮 Active Games</div>
                <div class="game-list">
    """
    
    if games:
        for game_id, game in games.items():
            state = game.get_state()
            html += f"""
                    <div class="game-item">
                        <a href="/games/{game_id}/view" class="game-link">{game_id}</a>
                        <div class="game-stats">
                            <div class="stat">
                                <span class="stat-label">Status</span>
                                <span class="stat-value">
                                    <span class="status-badge status-{state['status']}">{state['status']}</span>
                                </span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Score</span>
                                <span class="stat-value">{state['score']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Lines</span>
                                <span class="stat-value">{state['lines_cleared']}</span>
                            </div>
                        </div>
                    </div>
            """
    else:
        html += '<p style="text-align: center; color: #666;">No active games. Create a game using the API or start a replay above.</p>'
    
    html += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


@app.get("/games/{game_id}/view", response_class=HTMLResponse)
async def view_game(game_id: str, replay: Optional[str] = Query(None)):
    """View the current game board as HTML.
    
    Args:
        game_id: Game identifier
        replay: If "true", this is a replay view (deprecated, WebSocket handles this now)
        
    Returns:
        HTML page with visual representation of the board
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    state = game.get_state()
    board = state["board"]
    current_piece = state["current_piece"]
    next_piece = state["next_piece"]
    
    # Check if this is a replay
    is_replay = game_id in replay_games
    replay_info = None
    if is_replay:
        replay = replay_games[game_id]
        training_data = replay["training_data"]
        current_index = replay["current_move_index"]
        replay_info = {
            "move_index": current_index,
            "total_moves": len(training_data["moves"]),
            "is_complete": current_index >= len(training_data["moves"]),
            "filename": replay["filename"],
        }
    
    # Create HTML visualization
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PuzzleGrid Game - {game_id}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            .container {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 1200px;
                width: 100%;
            }}
            h1 {{
                color: #333;
                margin-top: 0;
                text-align: center;
            }}
            .game-info {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .info-box {{
                background: #f5f5f5;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .info-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
                margin-bottom: 5px;
            }}
            .info-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .game-area {{
                display: flex;
                gap: 30px;
                justify-content: center;
                flex-wrap: wrap;
            }}
            .board-container {{
                background: #1a1a1a;
                padding: 20px;
                border-radius: 10px;
                box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
            }}
            .board {{
                display: grid;
                grid-template-columns: repeat({len(board[0])}, 30px);
                grid-template-rows: repeat({len(board)}, 30px);
                gap: 2px;
                background: #000;
                padding: 5px;
                border-radius: 5px;
            }}
            .cell {{
                width: 30px;
                height: 30px;
                border-radius: 3px;
                background: #222;
                border: 1px solid #333;
            }}
            .cell.filled {{
                background: #4a90e2;
                border: 1px solid #5aa0f2;
                box-shadow: inset 0 0 5px rgba(255,255,255,0.3);
            }}
            .cell.current-piece {{
                background: #f39c12;
                border: 1px solid #f5b041;
                box-shadow: 0 0 10px rgba(243, 156, 18, 0.5);
            }}
            .cell.potential-placement {{
                outline: 2px solid rgba(155, 89, 182, 0.95);
                outline-offset: -3px;
                box-shadow: inset 0 0 9px rgba(155, 89, 182, 0.7);
            }}
            .cell.potential-placement:not(.filled):not(.current-piece) {{
                background: #3b255f;
            }}
            .placement-toggle {{
                width: 100%;
                margin-top: 15px;
                padding: 10px 12px;
                border: none;
                border-radius: 6px;
                background: #8e44ad;
                color: white;
                font-weight: bold;
                cursor: pointer;
                transition: background 0.2s;
            }}
            .placement-toggle:hover {{
                background: #7d3c98;
            }}
            .placement-toggle.active {{
                background: #2ecc71;
            }}
            .placements-panel {{
                display: none;
                margin-top: 12px;
                padding: 10px;
                background: #f2e8f7;
                border-left: 4px solid #8e44ad;
                border-radius: 4px;
                text-align: left;
            }}
            .placements-panel.visible {{
                display: block;
            }}
            .placements-count {{
                margin-bottom: 8px;
                color: #5b2c6f;
                font-size: 12px;
                font-weight: bold;
            }}
            .placements-list {{
                max-height: 180px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                color: #333;
            }}
            .placement-row {{
                padding: 3px 0;
                border-bottom: 1px solid rgba(142, 68, 173, 0.15);
            }}
            .next-piece-container {{
                background: #f5f5f5;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .next-piece-label {{
                font-size: 14px;
                color: #666;
                margin-bottom: 15px;
                font-weight: bold;
            }}
            .next-piece-preview {{
                display: inline-grid;
                grid-template-columns: repeat(4, 25px);
                grid-template-rows: repeat(4, 25px);
                gap: 2px;
                background: #fff;
                padding: 10px;
                border-radius: 5px;
            }}
            .next-cell {{
                width: 25px;
                height: 25px;
                border-radius: 2px;
                background: #eee;
            }}
            .next-cell.filled {{
                background: #4a90e2;
                border: 1px solid #5aa0f2;
            }}
            .status-badge {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
            }}
            .status-playing {{
                background: #2ecc71;
                color: white;
            }}
            .status-game-over {{
                background: #e74c3c;
                color: white;
            }}
            .status-paused {{
                background: #f39c12;
                color: white;
            }}
            .allowed-pieces {{
                margin-top: 20px;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 8px;
            }}
            .allowed-pieces-label {{
                font-size: 12px;
                color: #666;
                margin-bottom: 10px;
            }}
            .piece-tags {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }}
            .piece-tag {{
                background: #4a90e2;
                color: white;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
            }}
            .replay-info {{
                margin-bottom: 20px;
            }}
        </style>
        <script>
            const gameId = '{game_id}';
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            let showPotentialPlacements = false;
            let potentialPlacements = [];
            let latestGameData = null;
            
            function getPotentialCellMap(board) {{
                const potentialCellMap = new Map();
                if (!showPotentialPlacements) {{
                    return potentialCellMap;
                }}
                
                potentialPlacements.forEach(function(placement) {{
                    placement.cells.forEach(function(cell) {{
                        if (cell[0] >= 0 && cell[0] < board.length && cell[1] >= 0 && cell[1] < board[0].length) {{
                            const cellKey = cell[0] + ',' + cell[1];
                            if (!potentialCellMap.has(cellKey)) {{
                                potentialCellMap.set(cellKey, []);
                            }}
                            potentialCellMap.get(cellKey).push('#' + placement.index + ' r' + placement.row + ' c' + placement.col + ' rot' + placement.rotation);
                        }}
                    }});
                }});
                
                return potentialCellMap;
            }}
            
            function updatePlacementsPanel() {{
                const panel = document.getElementById('placementsPanel');
                const count = document.getElementById('placementsCount');
                const list = document.getElementById('placementsList');
                if (!panel || !count || !list) return;
                
                panel.classList.toggle('visible', showPotentialPlacements);
                count.textContent = potentialPlacements.length + ' potential placement' + (potentialPlacements.length === 1 ? '' : 's');
                list.innerHTML = '';
                
                potentialPlacements.forEach(function(placement) {{
                    const row = document.createElement('div');
                    row.className = 'placement-row';
                    row.textContent = '#' + placement.index + ' row=' + placement.row + ' col=' + placement.col + ' rot=' + placement.rotation;
                    list.appendChild(row);
                }});
                
                if (showPotentialPlacements && potentialPlacements.length === 0) {{
                    list.innerHTML = '<div class="placement-row">No legal placements for the current piece.</div>';
                }}
            }}
            
            async function refreshPotentialPlacements() {{
                if (!showPotentialPlacements) {{
                    potentialPlacements = [];
                    updatePlacementsPanel();
                    return;
                }}
                
                try {{
                    const response = await fetch('/games/' + encodeURIComponent(gameId) + '/potential-placements');
                    if (!response.ok) {{
                        throw new Error('Failed to load potential placements');
                    }}
                    const data = await response.json();
                    potentialPlacements = data.placements || [];
                }} catch (error) {{
                    console.error(error);
                    potentialPlacements = [];
                }}
                updatePlacementsPanel();
            }}
            
            function togglePotentialPlacements() {{
                showPotentialPlacements = !showPotentialPlacements;
                const button = document.getElementById('togglePotentialPlacements');
                if (button) {{
                    button.classList.toggle('active', showPotentialPlacements);
                    button.textContent = showPotentialPlacements ? 'Hide Potential Placements' : 'Show Potential Placements';
                }}
                
                refreshPotentialPlacements().then(function() {{
                    if (latestGameData) {{
                        updateGameState(latestGameData, true);
                    }}
                }});
            }}
            
            function updateGameState(data, skipPlacementRefresh = false) {{
                const {{ game_state, replay_info }} = data;
                
                if (!game_state) return;
                latestGameData = data;
                
                // Update game info boxes
                document.querySelector('.info-box:nth-child(1) .info-value').textContent = gameId;
                const statusEl = document.querySelector('.info-box:nth-child(2) .info-value');
                if (statusEl) {{
                    statusEl.innerHTML = '<span class="status-badge status-' + game_state.status + '">' + game_state.status + '</span>';
                }}
                document.querySelector('.info-box:nth-child(3) .info-value').textContent = game_state.score;
                document.querySelector('.info-box:nth-child(4) .info-value').textContent = game_state.lines_cleared;
                document.querySelector('.info-box:nth-child(5) .info-value').textContent = game_state.pieces_placed;
                document.querySelector('.info-box:nth-child(6) .info-value').textContent = 
                    game_state.board[0].length + '×' + game_state.board.length;
                
                // Update replay info if present
                if (replay_info) {{
                    const replayDiv = document.querySelector('.replay-info');
                    if (replayDiv) {{
                        const completeText = replay_info.is_complete ? 
                            ' | <span style="color: #2ecc71;">✓ Complete</span>' : '';
                        replayDiv.innerHTML = 
                            '<div style="font-weight: bold; color: #333; margin-bottom: 5px;">📼 Replay Mode</div>' +
                            '<div style="font-size: 14px; color: #666;">' +
                            'File: <strong>' + replay_info.filename + '</strong> | ' +
                            'Move: ' + replay_info.move_index + ' / ' + replay_info.total_moves + completeText +
                            '</div>';
                    }}
                }}
                
                // Update board
                const board = game_state.board;
                const currentPiece = game_state.current_piece;
                const boardElement = document.querySelector('.board');
                
                // Clear and rebuild board
                boardElement.innerHTML = '';
                boardElement.style.gridTemplateColumns = 'repeat(' + board[0].length + ', 30px)';
                boardElement.style.gridTemplateRows = 'repeat(' + board.length + ', 30px)';
                const potentialCellMap = getPotentialCellMap(board);
                
                // Get current piece cells (only if piece doesn't overlap with placed cells)
                const currentPieceCells = new Set();
                if (currentPiece && currentPiece.cells) {{
                    // Check if any cell of the current piece overlaps with placed cells
                    let hasOverlap = false;
                    currentPiece.cells.forEach(function(cell) {{
                        if (cell[0] >= 0 && cell[0] < board.length && cell[1] >= 0 && cell[1] < board[0].length) {{
                            // If the cell is already filled on the board, don't show it as current piece
                            if (board[cell[0]][cell[1]] !== 0) {{
                                hasOverlap = true;
                            }}
                        }}
                    }});
                    
                    // Only add current piece cells if there's no overlap (piece hasn't been placed yet)
                    if (!hasOverlap) {{
                        currentPiece.cells.forEach(function(cell) {{
                            if (cell[0] >= 0 && cell[0] < board.length && cell[1] >= 0 && cell[1] < board[0].length) {{
                                currentPieceCells.add(cell[0] + ',' + cell[1]);
                            }}
                        }});
                    }}
                }}
                
                // Render board cells
                for (let row = 0; row < board.length; row++) {{
                    for (let col = 0; col < board[row].length; col++) {{
                        const cell = document.createElement('div');
                        cell.className = 'cell';
                        const cellKey = row + ',' + col;
                        
                        if (currentPieceCells.has(cellKey)) {{
                            cell.classList.add('current-piece');
                        }} else if (board[row][col] !== 0) {{
                            cell.classList.add('filled');
                        }}
                        if (potentialCellMap.has(cellKey)) {{
                            cell.classList.add('potential-placement');
                            cell.title = 'Potential placements: ' + potentialCellMap.get(cellKey).join(', ');
                        }}
                        boardElement.appendChild(cell);
                    }}
                }}
                
                // Update current piece info
                if (currentPiece) {{
                    const currentPieceText = 'Type: <strong>' + currentPiece.type + '</strong><br>Position: (' + 
                        currentPiece.row + ', ' + currentPiece.col + ')<br>Rotation: ' + currentPiece.rotation;
                    const currentPieceDiv = document.querySelector('.next-piece-container .next-piece-label').nextElementSibling;
                    if (currentPieceDiv) {{
                        currentPieceDiv.innerHTML = currentPieceText;
                    }}
                }}
                
                // Update next piece text
                if (game_state.next_piece) {{
                    // Find the next piece label and update the text below it
                    const nextPieceLabels = document.querySelectorAll('.next-piece-label');
                    if (nextPieceLabels.length >= 2) {{
                        const nextPieceLabel = nextPieceLabels[1]; // Second label is "Next Piece"
                        const nextPieceTextDiv = nextPieceLabel.nextElementSibling;
                        if (nextPieceTextDiv) {{
                            nextPieceTextDiv.innerHTML = 'Type: <strong>' + game_state.next_piece + '</strong>';
                        }}
                    }}
                }}
                
                // Update last placed piece info
                if (game_state.last_placed_piece) {{
                    const lastPlaced = game_state.last_placed_piece;
                    const lastPlacedText = 'Type: <strong>' + lastPlaced.type + '</strong><br>Position: (' + 
                        lastPlaced.row + ', ' + lastPlaced.col + ')<br>Rotation: ' + lastPlaced.rotation;
                    const nextPieceLabels = document.querySelectorAll('.next-piece-label');
                    if (nextPieceLabels.length >= 3) {{
                        const lastPlacedLabel = nextPieceLabels[2]; // Third label is "Last Placed Piece"
                        const lastPlacedTextDiv = lastPlacedLabel.nextElementSibling;
                        if (lastPlacedTextDiv) {{
                            lastPlacedTextDiv.innerHTML = lastPlacedText;
                        }}
                    }}
                }}
                
                // Update floating pieces warning
                const warningDiv = document.querySelector('.next-piece-container > div[style*="background: #ffe6e6"]');
                if (game_state.has_floating_pieces) {{
                    if (!warningDiv) {{
                        const lastPlacedLabels = document.querySelectorAll('.next-piece-label');
                        if (lastPlacedLabels.length >= 3) {{
                            const lastPlacedLabel = lastPlacedLabels[2];
                            const warningHTML = '<div style="margin-top: 20px; padding: 10px; background: #ffe6e6; border-left: 4px solid #e74c3c; border-radius: 4px;"><div style="font-weight: bold; color: #e74c3c; margin-bottom: 5px;">⚠ Warning</div><div style="font-size: 11px; color: #c0392b;">Floating pieces detected! Board state may be invalid.</div></div>';
                            lastPlacedLabel.parentElement.insertAdjacentHTML('beforeend', warningHTML);
                        }}
                    }}
                }} else {{
                    if (warningDiv) {{
                        warningDiv.remove();
                    }}
                }}
                
                if (showPotentialPlacements && !skipPlacementRefresh) {{
                    refreshPotentialPlacements().then(function() {{
                        if (latestGameData) {{
                            updateGameState(latestGameData, true);
                        }}
                    }});
                }} else {{
                    updatePlacementsPanel();
                }}
            }}
            
            function connectWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/ws/games/' + gameId;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {{
                    console.log('WebSocket connected');
                    reconnectAttempts = 0;
                    const statusEl = document.getElementById('wsStatus');
                    if (statusEl) {{
                        statusEl.textContent = 'Connected via WebSocket';
                        statusEl.style.color = '#2ecc71';
                    }}
                }};
                
                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    if (data.type === 'game_update' || data.game_state) {{
                        updateGameState(data);
                    }}
                }};
                
                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    const statusEl = document.getElementById('wsStatus');
                    if (statusEl) {{
                        statusEl.textContent = 'WebSocket error';
                        statusEl.style.color = '#e74c3c';
                    }}
                }};
                
                ws.onclose = () => {{
                    console.log('WebSocket closed');
                    const statusEl = document.getElementById('wsStatus');
                    if (statusEl) {{
                        statusEl.textContent = 'WebSocket disconnected - reconnecting...';
                        statusEl.style.color = '#f39c12';
                    }}
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {{
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 1000 * reconnectAttempts);
                    }} else {{
                        if (statusEl) {{
                            statusEl.textContent = 'WebSocket connection failed';
                            statusEl.style.color = '#e74c3c';
                        }}
                    }}
                }};
            }}
            
            // Connect on page load
            window.addEventListener('DOMContentLoaded', () => {{
                const placementToggle = document.getElementById('togglePotentialPlacements');
                if (placementToggle) {{
                    placementToggle.addEventListener('click', togglePotentialPlacements);
                }}
                connectWebSocket();
            }});
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {{
                if (ws) {{
                    ws.close();
                }}
            }});
        </script>
    """
    
    html += f"""
    </head>
    <body>
        <div class="container">
            <h1>🎮 PuzzleGrid Game Viewer</h1>
    """
    
    # Add replay info div (will be updated via WebSocket)
    if is_replay and replay_info:
        complete_text = ' | <span style="color: #2ecc71;">✓ Complete</span>' if replay_info['is_complete'] else ''
        html += f"""
            <div class="replay-info" style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4a90e2;">
                <div style="font-weight: bold; color: #333; margin-bottom: 5px;">📼 Replay Mode</div>
                <div style="font-size: 14px; color: #666;">
                    File: <strong>{replay_info['filename']}</strong> | 
                    Move: {replay_info['move_index']} / {replay_info['total_moves']}{complete_text}
                </div>
            </div>
        """
    
    # Extract values for f-string formatting
    game_status = state['status']
    game_score = state['score']
    lines_cleared = state['lines_cleared']
    pieces_placed = state['pieces_placed']
    board_width = len(board[0])
    board_height = len(board)
    
    html += f"""
            <div class="game-info">
                <div class="info-box">
                    <div class="info-label">Game ID</div>
                    <div class="info-value">{game_id}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Status</div>
                    <div class="info-value">
                        <span class="status-badge status-{game_status}">{game_status}</span>
                    </div>
                </div>
                <div class="info-box">
                    <div class="info-label">Score</div>
                    <div class="info-value">{game_score}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Lines Cleared</div>
                    <div class="info-value">{lines_cleared}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Pieces Placed</div>
                    <div class="info-value">{pieces_placed}</div>
                </div>
                <div class="info-box">
                    <div class="info-label">Board Size</div>
                    <div class="info-value">{board_width}×{board_height}</div>
                </div>
            </div>
            
            <div class="game-area">
                <div class="board-container">
                    <div class="board">
    """
    
    # Get current piece cells for highlighting (only if piece doesn't overlap with placed cells)
    current_piece_cells = set()
    current_piece_info = None
    if current_piece and "cells" in current_piece:
        # Check if any cell of the current piece overlaps with placed cells
        has_overlap = False
        for cell in current_piece["cells"]:
            # Ensure cells are within board bounds
            if 0 <= cell[0] < len(board) and 0 <= cell[1] < len(board[0]):
                # If the cell is already filled on the board, don't show it as current piece
                if board[cell[0]][cell[1]] != 0:
                    has_overlap = True
                    break
        
        # Only add current piece cells if there's no overlap (piece hasn't been placed yet)
        if not has_overlap:
            for cell in current_piece["cells"]:
                # Ensure cells are within board bounds
                if 0 <= cell[0] < len(board) and 0 <= cell[1] < len(board[0]):
                    current_piece_cells.add((cell[0], cell[1]))
        
        current_piece_info = {
            "type": current_piece.get("type", "Unknown"),
            "row": current_piece.get("row", 0),
            "col": current_piece.get("col", 0),
            "rotation": current_piece.get("rotation", 0),
            "cell_count": len(current_piece_cells)
        }
    
    # Render board
    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            is_filled = cell != 0
            is_current = (row_idx, col_idx) in current_piece_cells
            
            # Current piece takes priority over filled cells
            if is_current:
                html += '<div class="cell current-piece"></div>'
            elif is_filled:
                html += '<div class="cell filled"></div>'
            else:
                html += '<div class="cell"></div>'
    
    # Format current piece info
    if current_piece_info:
        current_piece_text = f'Type: <strong>{current_piece_info["type"]}</strong><br>Position: ({current_piece_info["row"]}, {current_piece_info["col"]})<br>Rotation: {current_piece_info["rotation"]}'
    else:
        current_piece_text = 'None'
    
    html += f"""
                    </div>
                </div>
                
                <div class="next-piece-container">
                    <div class="next-piece-label">Current Piece</div>
                    <div style="margin-bottom: 15px; font-size: 12px; color: #666;">
                        {current_piece_text}
                    </div>
                    <button id="togglePotentialPlacements" class="placement-toggle" type="button">
                        Show Potential Placements
                    </button>
                    <div id="placementsPanel" class="placements-panel">
                        <div id="placementsCount" class="placements-count">0 potential placements</div>
                        <div id="placementsList" class="placements-list"></div>
                    </div>
                    <div class="next-piece-label" style="margin-top: 20px;">Next Piece</div>
                    <div style="margin-top: 15px; font-size: 12px; color: #666;">
                        Type: <strong>""" + (next_piece if next_piece else "N/A") + """</strong>
                    </div>
    """
    
    # Format last placed piece info
    last_placed_piece = state.get('last_placed_piece')
    if last_placed_piece:
        last_placed_text = f'Type: <strong>{last_placed_piece["type"]}</strong><br>Position: ({last_placed_piece["row"]}, {last_placed_piece["col"]})<br>Rotation: {last_placed_piece["rotation"]}'
    else:
        last_placed_text = 'None'
    
    html += f"""
                    <div class="next-piece-label" style="margin-top: 20px;">Last Placed Piece</div>
                    <div style="margin-top: 15px; font-size: 12px; color: #666;">
                        {last_placed_text}
                    </div>
    """
    
    # Add floating pieces warning if detected
    has_floating = state.get('has_floating_pieces', False)
    if has_floating:
        html += """
                    <div style="margin-top: 20px; padding: 10px; background: #ffe6e6; border-left: 4px solid #e74c3c; border-radius: 4px;">
                        <div style="font-weight: bold; color: #e74c3c; margin-bottom: 5px;">⚠ Warning</div>
                        <div style="font-size: 11px; color: #c0392b;">Floating pieces detected! Board state may be invalid.</div>
                    </div>
        """
    
    html += """
                </div>
            </div>
            
            <div class="allowed-pieces">
                <div class="allowed-pieces-label">Allowed Pieces</div>
                <div class="piece-tags">
    """
    
    for piece in state['allowed_pieces']:
        html += f'<span class="piece-tag">{piece}</span>'
    
    html += """
                </div>
            </div>
            
            <div style="margin-top: 20px; text-align: center; color: #666; font-size: 12px;">
                <p>API Endpoint: <code>/games/{game_id}</code></p>
                <p>WebSocket: <code>/ws/games/{game_id}</code></p>
                <p id="wsStatus" style="color: #2ecc71;">Connected via WebSocket</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)


@app.get("/training-data")
async def list_training_data():
    """List all available training data files.
    
    Returns:
        List of training data file names
    """
    training_dir = get_training_data_dir()
    if not os.path.exists(training_dir):
        return {"files": []}
    
    files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
    files.sort(reverse=True)  # Most recent first
    return {"files": files}


@app.post("/replay/start")
async def start_replay(filename: str):
    """Start replaying a training data file.
    
    Args:
        filename: Name of the training data file (from training_data directory)
        
    Returns:
        Game ID for the replay
    """
    training_dir = get_training_data_dir()
    filepath = os.path.join(training_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Training data file not found: {filename}")
    
    try:
        with open(filepath, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading training data: {str(e)}")
    
    # Validate training data structure
    if "moves" not in training_data or not training_data["moves"]:
        raise HTTPException(status_code=400, detail="Training data has no moves")
    
    # Create a new game with the same configuration as the training data
    allowed_pieces = None
    if "allowed_pieces" in training_data and training_data["allowed_pieces"]:
        try:
            allowed_pieces = [PieceType[p.upper()] for p in training_data["allowed_pieces"]]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid piece type in training data: {e}")
    
    # Extract piece sequence for deterministic replay
    piece_sequence = training_data.get("piece_sequence")
    
    game = PuzzleGridGame(
        width=training_data.get("width", 10),
        height=training_data.get("height", 20),
        allowed_pieces=allowed_pieces,
        piece_sequence=piece_sequence,  # Use the recorded piece sequence for deterministic replay
        game_id=f"replay_{filename.replace('.json', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Store replay state
    replay_games[game.game_id] = {
        "training_data": training_data,
        "current_move_index": 0,
        "game_instance": game,
        "filename": filename,
    }
    
    # Also add to games dict so it can be viewed normally
    games[game.game_id] = game
    
    # Start background task for automatic replay advancement
    task = asyncio.create_task(replay_advancement_task(game.game_id))
    replay_tasks[game.game_id] = task
    
    return {"game_id": game.game_id, "total_moves": len(training_data["moves"])}


@app.get("/replay/{game_id}/next")
async def replay_next_move(game_id: str):
    """Advance the replay by one move.
    
    Args:
        game_id: Replay game ID
        
    Returns:
        Updated game state and move info
    """
    if game_id not in replay_games:
        raise HTTPException(status_code=404, detail="Replay not found")
    
    replay = replay_games[game_id]
    training_data = replay["training_data"]
    current_index = replay["current_move_index"]
    game = replay["game_instance"]
    
    if current_index >= len(training_data["moves"]):
        return {
            "game_state": game.get_state(),
            "replay_info": {
                "move_index": current_index,
                "total_moves": len(training_data["moves"]),
                "is_complete": True,
                "filename": replay["filename"],
            },
        }
    
    # Get the next move
    move_data = training_data["moves"][current_index]
    
    # Apply the move
    if move_data["move_type"] == "place":
        pos = move_data["piece_position"]
        
        # Verify that the current piece matches what's expected from the move data
        # This helps catch issues with piece sequence mismatches
        expected_piece_type = move_data.get("piece_type")
        if expected_piece_type and game.current_piece:
            current_piece_type = game.current_piece.piece_type.value
            if current_piece_type.upper() != expected_piece_type.upper():
                # Log warning but continue - this shouldn't happen with correct piece_sequence
                print(f"Warning: Piece type mismatch at move {current_index}: expected {expected_piece_type}, got {current_piece_type}")
        
        # For replay, place the piece directly - drop_to_rest handles falling automatically
        from .pieces import Piece
        if game.current_piece is None:
            # If no current piece, try to get from move data
            from .pieces import PieceType
            piece_type_str = move_data.get("piece_type")
            if piece_type_str:
                piece_type = PieceType[piece_type_str.upper()]
            else:
                # Fallback: just place normally
                success = game.place_piece_direct(
                    row=pos["row"],
                    col=pos["col"],
                    rotation=pos.get("rotation", 0),
                    drop_to_rest=True,
                )
                if not success:
                    pass
        else:
            # For replay, just place the piece directly - drop_to_rest will handle falling automatically
            # This is much faster and avoids blocking the replay advancement
            success = game.place_piece_direct(
                row=pos["row"],
                col=pos["col"],
                rotation=pos.get("rotation", 0),
                drop_to_rest=True,  # Automatically drops to resting position
            )
            
            if not success:
                # If placement fails, try to continue anyway
                pass
    else:
        # For other move types, try to apply them
        move_mapping = {mt.value: mt for mt in MoveType}
        move_type = move_mapping.get(move_data["move_type"])
        if move_type:
            game.move(move_type)
    
    # Advance to next move
    replay["current_move_index"] = current_index + 1
    
    result = {
        "game_state": game.get_state(),
        "replay_info": {
            "move_index": current_index + 1,
            "total_moves": len(training_data["moves"]),
            "is_complete": current_index + 1 >= len(training_data["moves"]),
            "filename": replay["filename"],
        },
        "current_move": move_data,
    }
    
    # Broadcast update to WebSocket clients
    await broadcast_game_update(game_id, {
        "type": "game_update",
        **result
    })
    
    return result


@app.get("/replay/{game_id}/state")
async def get_replay_state(game_id: str):
    """Get current replay state.
    
    Args:
        game_id: Replay game ID
        
    Returns:
        Current replay state
    """
    if game_id not in replay_games:
        raise HTTPException(status_code=404, detail="Replay not found")
    
    replay = replay_games[game_id]
    training_data = replay["training_data"]
    current_index = replay["current_move_index"]
    game = replay["game_instance"]
    
    return {
        "game_state": game.get_state(),
        "move_index": current_index,
        "total_moves": len(training_data["moves"]),
        "is_complete": current_index >= len(training_data["moves"]),
        "filename": replay["filename"],
    }


@app.get("/games/{game_id}/state")
async def get_game_state_api(game_id: str):
    """Get current game state as JSON (for WebSocket updates).
    
    Args:
        game_id: Game identifier
        
    Returns:
        Game state with replay info if applicable
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    state = game.get_state()
    
    result = {"game_state": state}
    
    # Add replay info if this is a replay
    if game_id in replay_games:
        replay = replay_games[game_id]
        training_data = replay["training_data"]
        current_index = replay["current_move_index"]
        result["replay_info"] = {
            "move_index": current_index,
            "total_moves": len(training_data["moves"]),
            "is_complete": current_index >= len(training_data["moves"]),
            "filename": replay["filename"],
        }
    
    return result


@app.websocket("/ws/games/{game_id}")
async def websocket_game_updates(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates.
    
    Args:
        websocket: WebSocket connection
        game_id: Game identifier
    """
    await websocket.accept()
    
    # Add to connections
    if game_id not in websocket_connections:
        websocket_connections[game_id] = set()
    websocket_connections[game_id].add(websocket)
    
    try:
        # Send initial state
        if game_id in games:
            game = games[game_id]
            state = game.get_state()
            result = {"game_state": state}
            
            if game_id in replay_games:
                replay = replay_games[game_id]
                training_data = replay["training_data"]
                current_index = replay["current_move_index"]
                result["replay_info"] = {
                    "move_index": current_index,
                    "total_moves": len(training_data["moves"]),
                    "is_complete": current_index >= len(training_data["moves"]),
                    "filename": replay["filename"],
                }
            
            await websocket.send_json(result)
        
        # Keep connection alive and wait for messages
        while True:
            try:
                # Wait for ping or close
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        # Remove from connections
        if game_id in websocket_connections:
            websocket_connections[game_id].discard(websocket)
            if not websocket_connections[game_id]:
                del websocket_connections[game_id]


async def broadcast_game_update(game_id: str, data: dict):
    """Broadcast game update to all connected WebSocket clients.
    
    Args:
        game_id: Game identifier
        data: Data to broadcast
    """
    if game_id not in websocket_connections:
        return
    
    disconnected = set()
    for websocket in websocket_connections[game_id]:
        try:
            await websocket.send_json(data)
        except Exception:
            disconnected.add(websocket)
    
    # Remove disconnected websockets
    for ws in disconnected:
        websocket_connections[game_id].discard(ws)
    
    if not websocket_connections[game_id]:
        del websocket_connections[game_id]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PuzzleGrid API",
        "version": "0.1.0",
        "description": "Puzzle grid game API for reinforcement learning",
        "endpoints": {
            "GET /view": "View list of all games (HTML)",
            "GET /games/{game_id}/view": "View game board as HTML",
            "POST /games": "Create a new game",
            "GET /games": "List all games",
            "GET /games/{game_id}": "Get game state",
            "POST /games/{game_id}/move": "Make a move",
            "POST /games/{game_id}/place": "Place piece directly",
            "GET /games/{game_id}/moves": "Get move history as JSON",
            "GET /games/{game_id}/potential-placements": "Get legal placements for the current piece",
            "GET /games/{game_id}/download": "Download moves as JSON file",
            "DELETE /games/{game_id}": "Delete a game",
            "GET /training-data": "List available training data files",
            "POST /replay/start": "Start replaying a training data file",
            "GET /replay/{game_id}/next": "Advance replay by one move",
            "GET /replay/{game_id}/state": "Get current replay state",
        }
    }

