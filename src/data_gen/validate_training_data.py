#!/usr/bin/env python3
"""Validate training data by replaying games and checking lines cleared.

This script:
- Loads all training data files
- Extracts expected lines cleared from filename and metadata
- Replays each game using the API
- Compares actual lines cleared with expected
- Reports mismatches, timeouts, and failures
"""

import json
import os
import re
import sys
import time
import requests
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 120  # seconds per game replay
DEFAULT_DATA_DIR = "./data/training_data_clean"


class TrainingDataValidator:
    """Validates training data by replaying games."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, api_url: str = BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        """Initialize validator.
        
        Args:
            data_dir: Directory containing training data files
            api_url: Base URL of the PuzzleGrid API
            timeout: Timeout in seconds for each replay
        """
        self.data_dir = data_dir
        self.api_url = api_url
        self.timeout = timeout
        self.results = []
    
    def extract_lines_from_filename(self, filename: str) -> Optional[int]:
        """Extract lines cleared from filename.
        
        Args:
            filename: Training data filename (e.g., "I_O_Z_12lines_20260110_162307.json")
            
        Returns:
            Number of lines cleared or None if not found
        """
        match = re.search(r'(\d+)lines', filename)
        if match:
            return int(match.group(1))
        return None
    
    def load_training_file(self, filepath: str) -> Optional[Dict]:
        """Load a training data file.
        
        Args:
            filepath: Path to training data file
            
        Returns:
            Training data dictionary or None if error
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ERROR: Failed to load {filepath}: {e}")
            return None
    
    def get_expected_lines(self, filename: str, data: Dict) -> Tuple[Optional[int], Optional[int]]:
        """Get expected lines cleared from filename and metadata.
        
        Args:
            filename: Training data filename
            data: Training data dictionary
            
        Returns:
            Tuple of (filename_lines, metadata_lines)
        """
        filename_lines = self.extract_lines_from_filename(filename)
        metadata_lines = data.get("final_lines_cleared")
        return (filename_lines, metadata_lines)
    
    def replay_game(self, data: Dict) -> Tuple[bool, Optional[int], Optional[str]]:
        """Replay a game using the API.
        
        Args:
            data: Training data dictionary
            
        Returns:
            Tuple of (success, lines_cleared, error_message)
        """
        try:
            # Create game from training data
            allowed_pieces = data.get("allowed_pieces")
            piece_sequence = data.get("piece_sequence")
            
            # Create game with piece sequence for deterministic replay
            create_response = requests.post(
                f"{self.api_url}/games",
                json={
                    "width": data.get("width", 10),
                    "height": data.get("height", 20),
                    "allowed_pieces": allowed_pieces if allowed_pieces else None,
                    "piece_sequence": piece_sequence,  # Use recorded piece sequence
                },
                timeout=10
            )
            create_response.raise_for_status()
            game_data = create_response.json()
            game_id = game_data["game_id"]
            
            # Apply all moves
            moves = data.get("moves", [])
            start_time = time.time()
            
            for move_idx, move in enumerate(moves):
                # Check timeout
                if time.time() - start_time > self.timeout:
                    # Clean up game
                    try:
                        requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
                    except:
                        pass
                    return (False, None, f"Timeout after {self.timeout}s at move {move_idx}/{len(moves)}")
                
                if move["move_type"] == "place":
                    pos = move["piece_position"]
                    expected_piece_type = move.get("piece_type")
                    is_last_move = (move_idx == len(moves) - 1)
                    
                    # Check current piece type if piece_type is recorded in training data
                    if expected_piece_type:
                        # Get current game state to check piece type
                        state_response = requests.get(f"{self.api_url}/games/{game_id}", timeout=10)
                        state_response.raise_for_status()
                        current_state = state_response.json()
                        current_piece = current_state.get("current_piece")
                        
                        if current_piece:
                            current_piece_type = current_piece.get("type")
                            if current_piece_type and current_piece_type.upper() != expected_piece_type.upper():
                                # Clean up game
                                try:
                                    requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
                                except:
                                    pass
                                return (False, None, f"Piece type mismatch at move {move_idx}: expected {expected_piece_type}, got {current_piece_type}")
                    
                    # Training data records the FINAL position after dropping
                    # Place at that exact position WITHOUT dropping (drop_to_rest=False)
                    # because the position is already the final resting position
                    place_response = requests.post(
                        f"{self.api_url}/games/{game_id}/place",
                        json={
                            "row": pos["row"],
                            "col": pos["col"],
                            "rotation": pos.get("rotation", 0),
                            "drop_to_rest": False,  # Place exactly at recorded position
                        },
                        timeout=10
                    )
                    if place_response.status_code != 200:
                        # If this is the last move and placement fails, check if game would end anyway
                        # This handles the case where a piece spawns overlapping existing pieces (game over)
                        if is_last_move:
                            # Check if game is already over or would end
                            state_response = requests.get(f"{self.api_url}/games/{game_id}", timeout=10)
                            state_response.raise_for_status()
                            final_state = state_response.json()
                            
                            # If game is over, this is valid - the piece couldn't be placed, game ends
                            if final_state.get("status") != "playing":
                                # Game ended, which is expected for the last move if placement fails
                                # Continue to final state check
                                pass
                            else:
                                # Game is still playing but placement failed - this is an error
                                try:
                                    requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
                                except:
                                    pass
                                error_detail = place_response.json().get("detail", "Unknown error")
                                return (False, None, f"Placement failed at last move {move_idx}: {error_detail}")
                        else:
                            # Not the last move - this is a real error
                            try:
                                requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
                            except:
                                pass
                            error_detail = place_response.json().get("detail", "Unknown error")
                            return (False, None, f"Placement failed at move {move_idx}: {error_detail}")
                else:
                    # Other move types
                    move_response = requests.post(
                        f"{self.api_url}/games/{game_id}/move",
                        json={"move": move["move_type"]},
                        timeout=10
                    )
                    if move_response.status_code != 200:
                        # Clean up game
                        try:
                            requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
                        except:
                            pass
                        error_detail = move_response.json().get("detail", "Unknown error")
                        return (False, None, f"Move failed at move {move_idx}: {error_detail}")
            
            # Get final state
            final_response = requests.get(f"{self.api_url}/games/{game_id}", timeout=10)
            final_response.raise_for_status()
            final_state = final_response.json()
            lines_cleared = final_state.get("lines_cleared", 0)
            
            # Clean up game
            try:
                requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
            except:
                pass
            
            return (True, lines_cleared, None)
            
        except requests.exceptions.Timeout:
            # Try to clean up
            try:
                if 'game_id' in locals():
                    requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
            except:
                pass
            return (False, None, f"Request timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            # Try to clean up
            try:
                if 'game_id' in locals():
                    requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
            except:
                pass
            return (False, None, f"API error: {str(e)}")
        except Exception as e:
            # Try to clean up
            try:
                if 'game_id' in locals():
                    requests.delete(f"{self.api_url}/games/{game_id}", timeout=5)
            except:
                pass
            return (False, None, f"Unexpected error: {str(e)}")
    
    def validate_file(self, filepath: str) -> Dict:
        """Validate a single training data file.
        
        Args:
            filepath: Path to training data file
            
        Returns:
            Validation result dictionary
        """
        filename = os.path.basename(filepath)
        print(f"\nValidating: {filename}")
        
        # Load file
        data = self.load_training_file(filepath)
        if data is None:
            return {
                "filename": filename,
                "filepath": filepath,
                "status": "error",
                "error": "Failed to load file",
                "expected_filename": None,
                "expected_metadata": None,
                "actual_lines": None,
            }
        
        # Get expected lines
        expected_filename, expected_metadata = self.get_expected_lines(filename, data)
        
        # Check if filename and metadata match
        filename_metadata_match = (expected_filename == expected_metadata) if (expected_filename is not None and expected_metadata is not None) else None
        
        # Replay game
        print(f"  Expected (filename): {expected_filename}")
        print(f"  Expected (metadata): {expected_metadata}")
        print(f"  Replaying game...")
        success, actual_lines, error = self.replay_game(data)
        
        if not success:
            return {
                "filename": filename,
                "filepath": filepath,
                "status": "replay_failed",
                "error": error,
                "expected_filename": expected_filename,
                "expected_metadata": expected_metadata,
                "actual_lines": actual_lines,
                "filename_metadata_match": filename_metadata_match,
            }
        
        print(f"  Actual lines cleared: {actual_lines}")
        
        # Compare with expected
        matches_filename = (actual_lines == expected_filename) if expected_filename is not None else None
        matches_metadata = (actual_lines == expected_metadata) if expected_metadata is not None else None
        
        if matches_filename and matches_metadata:
            status = "valid"
            print(f"  ✓ VALID: Matches both filename and metadata")
        elif matches_metadata:
            status = "filename_mismatch"
            print(f"  ⚠ WARNING: Matches metadata but not filename (expected {expected_filename})")
        elif matches_filename:
            status = "metadata_mismatch"
            print(f"  ⚠ WARNING: Matches filename but not metadata (expected {expected_metadata})")
        else:
            status = "mismatch"
            print(f"  ✗ MISMATCH: Expected {expected_filename or expected_metadata}, got {actual_lines}")
        
        return {
            "filename": filename,
            "filepath": filepath,
            "status": status,
            "error": None,
            "expected_filename": expected_filename,
            "expected_metadata": expected_metadata,
            "actual_lines": actual_lines,
            "filename_metadata_match": filename_metadata_match,
            "matches_filename": matches_filename,
            "matches_metadata": matches_metadata,
        }
    
    def validate_all(self, max_files: Optional[int] = None) -> List[Dict]:
        """Validate all training data files.
        
        Args:
            max_files: Maximum number of files to validate (None for all)
            
        Returns:
            List of validation results
        """
        # Find all JSON files
        data_path = Path(self.data_dir)
        if not data_path.exists():
            print(f"ERROR: Data directory does not exist: {self.data_dir}")
            return []
        
        json_files = sorted(list(data_path.glob("*.json")))
        if max_files:
            json_files = json_files[:max_files]
        
        print(f"Found {len(json_files)} training data files")
        print(f"API URL: {self.api_url}")
        print(f"Timeout per game: {self.timeout}s")
        
        # Check API is available
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            response.raise_for_status()
        except Exception as e:
            print(f"\nERROR: Cannot connect to API at {self.api_url}: {e}")
            print("Make sure the API server is running!")
            return []
        
        # Validate each file
        results = []
        for i, filepath in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] ", end="")
            result = self.validate_file(str(filepath))
            results.append(result)
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print validation summary."""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total = len(self.results)
        valid = sum(1 for r in self.results if r["status"] == "valid")
        filename_mismatch = sum(1 for r in self.results if r["status"] == "filename_mismatch")
        metadata_mismatch = sum(1 for r in self.results if r["status"] == "metadata_mismatch")
        mismatch = sum(1 for r in self.results if r["status"] == "mismatch")
        replay_failed = sum(1 for r in self.results if r["status"] == "replay_failed")
        error = sum(1 for r in self.results if r["status"] == "error")
        
        print(f"\nTotal files: {total}")
        print(f"  ✓ Valid: {valid} ({valid/total*100:.1f}%)")
        print(f"  ⚠ Filename mismatch: {filename_mismatch}")
        print(f"  ⚠ Metadata mismatch: {metadata_mismatch}")
        print(f"  ✗ Mismatch (both): {mismatch}")
        print(f"  ✗ Replay failed: {replay_failed}")
        print(f"  ✗ Error: {error}")
        
        # Show details of failures
        if filename_mismatch + metadata_mismatch + mismatch + replay_failed + error > 0:
            print("\n" + "="*80)
            print("FAILURES AND WARNINGS")
            print("="*80)
            
            for result in self.results:
                if result["status"] != "valid":
                    print(f"\n{result['filename']}")
                    print(f"  Status: {result['status']}")
                    if result.get("error"):
                        print(f"  Error: {result['error']}")
                    if result.get("expected_filename") is not None:
                        print(f"  Expected (filename): {result['expected_filename']}")
                    if result.get("expected_metadata") is not None:
                        print(f"  Expected (metadata): {result['expected_metadata']}")
                    if result.get("actual_lines") is not None:
                        print(f"  Actual: {result['actual_lines']}")
        
        print("\n" + "="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate training data by replaying games"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing training data (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=BASE_URL,
        help=f"API base URL (default: {BASE_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per game in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to validate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results JSON (optional)"
    )
    
    args = parser.parse_args()
    
    validator = TrainingDataValidator(
        data_dir=args.data_dir,
        api_url=args.api_url,
        timeout=args.timeout
    )
    
    results = validator.validate_all(max_files=args.max_files)
    
    validator.print_summary()
    
    # Save results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(results),
            "results": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if there are failures
    failures = sum(1 for r in results if r["status"] not in ["valid", "filename_mismatch", "metadata_mismatch"])
    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

