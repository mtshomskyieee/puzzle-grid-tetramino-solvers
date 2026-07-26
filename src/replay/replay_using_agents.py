#!/usr/bin/env python3
"""Play puzzle grid with a LangGraph multi-agent client.

Agents (all in this file):
  - game_state_agent: read board / piece from the FastAPI engine
  - final_placement_agent: transformer scorer by default; one-shot LLM after a miss
  - movement_agent: incremental left/right/rotate/down/drop toward the target
  - validation_agent: check locked pose vs target; arm LLM assist for the next piece

Requires the API server (`uv run python main.py`) and OPENAI_API_KEY in `.env`
(only used when a prior placement missed).
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from puzzlegrid.board import Board
from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement
from replay_transformers import TransformerScorerPlayer

Placement = Tuple[int, int, int]
MoveName = Literal["left", "right", "rotate", "down", "drop"]

log = logging.getLogger("agent_replay")


def _fmt_pose(p: Optional[Placement]) -> str:
    if p is None:
        return "None"
    return f"(row={p[0]}, col={p[1]}, rot={p[2]})"


def configure_logging(level: str = "INFO") -> None:
    """Configure stdout logging for agent_replay (idempotent-ish)."""
    root = logging.getLogger("agent_replay")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
    # Keep library noise down unless user asked for DEBUG on our logger only.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

# Soft-drop into a resting non-target locks the piece early; only allow that edge
# when the resting state *is* the goal.
_MOVE_EDGES: Tuple[Tuple[MoveName, Any], ...] = (
    ("left", lambda p: p.move(0, -1)),
    ("right", lambda p: p.move(0, 1)),
    ("rotate", lambda p: p.rotate()),
    ("down", lambda p: p.move(1, 0)),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def board_to_ascii(board: List[List[int]]) -> str:
    lines = []
    for row in board:
        lines.append("".join("#" if c else "." for c in row))
    return "\n".join(lines)


def _rests(board: Board, piece: Piece) -> bool:
    return not board.is_valid_position(piece.move(1, 0).get_cells())


def reconstruct_path(
    board_state: List[List[int]],
    piece_type: PieceType,
    target: Placement,
) -> List[MoveName]:
    """BFS path from spawn to target that is safe for incremental /move.

    Soft-drop into a resting state locks in the engine, so those edges are only
    kept when the landing state is the target. Arriving at the target via
    left/right/rotate leaves the piece unlocked; the caller should then `drop`.
    """
    board = Board(width=len(board_state[0]), height=len(board_state))
    board.grid = [r[:] for r in board_state]
    target = (target[0], target[1], target[2] % 4)

    spawn = placement.get_spawn_state(piece_type, board.width)
    spawn_piece = Piece(piece_type, row=spawn[0], col=spawn[1], rotation=spawn[2])
    if not board.is_valid_position(spawn_piece.get_cells()):
        return []

    if spawn == target:
        return []  # already there; caller locks with drop

    parents: Dict[Placement, Tuple[Optional[Placement], Optional[MoveName]]] = {
        spawn: (None, None)
    }
    queue: deque[Placement] = deque([spawn])

    while queue:
        state = queue.popleft()
        row, col, rot = state
        piece = Piece(piece_type, row=row, col=col, rotation=rot)

        for move_name, apply in _MOVE_EDGES:
            nxt_piece = apply(piece)
            nxt = (nxt_piece.row, nxt_piece.col, nxt_piece.rotation % 4)
            if nxt in parents:
                continue
            if not board.is_valid_position(nxt_piece.get_cells()):
                continue

            landing_rests = _rests(board, nxt_piece)
            if move_name == "down" and landing_rests and nxt != target:
                # Would lock at the wrong placement — do not expand.
                continue

            parents[nxt] = (state, move_name)
            if nxt == target:
                # Reconstruct
                path: List[MoveName] = []
                cur: Optional[Placement] = nxt
                while cur is not None:
                    prev, mv = parents[cur]
                    if mv is not None:
                        path.append(mv)
                    cur = prev
                path.reverse()
                return path

            # Soft-drop onto target would lock; no further expansion needed (handled above).
            if move_name == "down" and landing_rests:
                continue

            queue.append(nxt)

    return []


class ChosenPlacement(BaseModel):
    row: int
    col: int
    rotation: int = Field(ge=0, le=3)
    reason: str = ""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentPlayState(TypedDict, total=False):
    game_id: str
    api_url: str
    board: List[List[int]]
    current_piece: Optional[dict]
    status: str
    score: int
    lines_cleared: int
    pieces_placed: int
    target: Optional[Placement]
    target_piece_type: Optional[str]
    llm_assist_once: bool
    failure_context: Optional[str]
    pieces_placed_before: int
    locked_pose: Optional[Placement]
    placement_ok: Optional[bool]
    move_count: int
    max_moves: int
    openai_model: str
    done: bool


# ---------------------------------------------------------------------------
# HTTP game client
# ---------------------------------------------------------------------------

class GameAPI:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.game_id: Optional[str] = None

    def create_game(self, allowed_pieces: Optional[List[str]] = None) -> dict:
        payload: Dict[str, Any] = {"width": 10, "height": 20}
        if allowed_pieces:
            payload["allowed_pieces"] = allowed_pieces
        r = requests.post(f"{self.api_url}/games", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        self.game_id = data["game_id"]
        return data

    def get_state(self) -> dict:
        r = requests.get(f"{self.api_url}/games/{self.game_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def move(self, move: MoveName) -> dict:
        r = requests.post(
            f"{self.api_url}/games/{self.game_id}/move",
            json={"move": move},
            timeout=10,
        )
        if r.status_code != 200:
            # Invalid move — return current state without raising so the agent can recover
            return self.get_state()
        return r.json()


# ---------------------------------------------------------------------------
# Agent nodes (closures over scorer + optional LLM)
# ---------------------------------------------------------------------------

def _apply_state_snapshot(state: AgentPlayState, st: dict) -> Dict[str, Any]:
    return {
        "board": st.get("board"),
        "current_piece": st.get("current_piece"),
        "status": st.get("status", "game_over"),
        "score": st.get("score", 0),
        "lines_cleared": st.get("lines_cleared", 0),
        "pieces_placed": st.get("pieces_placed", 0),
    }


def build_graph(
    api: GameAPI,
    scorer: TransformerScorerPlayer,
    openai_model: str = "gpt-4o-mini",
):
    llm: Optional[ChatOpenAI] = None

    def get_llm() -> ChatOpenAI:
        nonlocal llm
        if llm is None:
            llm = ChatOpenAI(model=openai_model, temperature=0)
        return llm

    def game_state_agent(state: AgentPlayState) -> Dict[str, Any]:
        st = api.get_state()
        out = _apply_state_snapshot(state, st)
        move_count = state.get("move_count", 0)
        max_moves = state.get("max_moves", 400)
        status = out["status"]
        done = (
            status != "playing"
            or not out.get("current_piece")
            or move_count >= max_moves
        )
        out["done"] = done
        cur = out.get("current_piece")
        piece_s = cur["type"] if cur else "-"
        log.info(
            "state  pieces=%s lines=%s score=%s status=%s next_piece=%s "
            "llm_assist_pending=%s turn=%s/%s%s",
            out.get("pieces_placed", 0),
            out.get("lines_cleared", 0),
            out.get("score", 0),
            status,
            piece_s,
            bool(state.get("llm_assist_once")),
            move_count,
            max_moves,
            " DONE" if done else "",
        )
        if state.get("failure_context") and state.get("llm_assist_once"):
            log.info("state  failure_context=%s", state["failure_context"])
        return out

    def final_placement_agent(state: AgentPlayState) -> Dict[str, Any]:
        board = state["board"]
        cur = state.get("current_piece")
        if not cur:
            log.warning("placement: no current piece — stopping")
            return {"target": None, "done": True}

        piece_type = PieceType[cur["type"]]
        candidates = placement.get_valid_placements(board, piece_type)
        if not candidates:
            log.warning("placement: piece=%s has 0 legal candidates — stopping",
                        piece_type.value)
            return {"target": None, "done": True}

        use_llm = bool(state.get("llm_assist_once"))
        n = state.get("pieces_placed", 0) + 1
        choice: Optional[Placement] = None
        source = "transformer"

        if use_llm:
            log.info(
                "placement #%s LLM ASSIST piece=%s candidates=%d prior_miss=%s",
                n,
                piece_type.value,
                len(candidates),
                state.get("failure_context") or "(none)",
            )
            choice, reason = _llm_choose_placement(
                get_llm(),
                board,
                piece_type,
                candidates,
                state.get("failure_context"),
            )
            if choice is None:
                source = "transformer_fallback"
                choice = scorer.choose(board, piece_type)
                log.warning(
                    "placement #%s LLM returned no legal pick — falling back to "
                    "transformer → %s",
                    n,
                    _fmt_pose(choice),
                )
            else:
                source = "llm"
                log.info(
                    "placement #%s LLM chose %s reason=%r",
                    n,
                    _fmt_pose(choice),
                    reason or "",
                )
            return {
                "target": choice,
                "target_piece_type": piece_type.value,
                "llm_assist_once": False,
                "failure_context": None,
                "pieces_placed_before": state.get("pieces_placed", 0),
                "locked_pose": None,
                "placement_ok": None,
            }

        choice = scorer.choose(board, piece_type)
        log.info(
            "placement #%s source=%s piece=%s target=%s candidates=%d",
            n,
            source,
            piece_type.value,
            _fmt_pose(choice),
            len(candidates),
        )
        return {
            "target": choice,
            "target_piece_type": piece_type.value if choice else None,
            "pieces_placed_before": state.get("pieces_placed", 0),
            "locked_pose": None,
            "placement_ok": None,
        }

    def movement_agent(state: AgentPlayState) -> Dict[str, Any]:
        target = state.get("target")
        board = state["board"]
        cur = state.get("current_piece")
        if not target or not cur:
            log.warning("movement: missing target or piece — skip")
            return {"locked_pose": None}

        piece_type = PieceType[cur["type"]]
        pieces_before = state.get("pieces_placed_before", state.get("pieces_placed", 0))
        path = reconstruct_path(board, piece_type, target)
        lock_via = "none"

        # If path empty but already at target (spawn == target), just drop to lock.
        st = api.get_state()
        pose = _piece_pose(st.get("current_piece"))
        if not path and pose == (target[0], target[1], target[2] % 4):
            path = []

        log.info(
            "movement piece=%s target=%s path_len=%d start=%s",
            piece_type.value,
            _fmt_pose(target),
            len(path),
            _fmt_pose(pose),
        )
        log.debug("movement path=%s", path)

        steps = 0
        max_steps = 200
        locked_pose: Optional[Placement] = None

        for mv in path:
            if steps >= max_steps:
                log.warning("movement: hit max_steps=%d mid-path", max_steps)
                break
            before = api.get_state()
            before_pose = _piece_pose(before.get("current_piece"))
            before_pp = before.get("pieces_placed", 0)
            st = api.move(mv)
            steps += 1
            after_pp = st.get("pieces_placed", 0)
            if after_pp > before_pp:
                locked_pose = _infer_locked_pose(before_pose, mv, board, piece_type)
                lock_via = f"path:{mv}"
                break

        # If still unlocked, finish with drop (at target) or LLM-assisted recovery.
        if locked_pose is None:
            st = api.get_state()
            if st.get("pieces_placed", 0) == pieces_before and st.get("current_piece"):
                pose = _piece_pose(st["current_piece"])
                tgt = (target[0], target[1], target[2] % 4)
                if pose == tgt:
                    st = api.move("drop")
                    if st.get("pieces_placed", 0) > pieces_before:
                        locked_pose = pose
                        lock_via = "drop_at_target"
                else:
                    why = "path_miss" if path else "no_bfs_path"
                    log.warning(
                        "movement %s — invoking LLM move recovery "
                        "(at=%s target=%s steps_used=%d)",
                        why,
                        _fmt_pose(pose),
                        _fmt_pose(tgt),
                        steps,
                    )
                    locked_pose = _llm_or_drop_recover(
                        api, get_llm, target, board, piece_type, max_steps - steps
                    )
                    lock_via = "llm_recover"

        st = api.get_state()
        out = _apply_state_snapshot(state, st)
        out["locked_pose"] = locked_pose
        out["move_count"] = state.get("move_count", 0) + 1
        log.info(
            "movement done via=%s locked=%s pieces_placed=%s→%s",
            lock_via,
            _fmt_pose(locked_pose),
            pieces_before,
            out.get("pieces_placed", 0),
        )
        return out

    def validation_agent(state: AgentPlayState) -> Dict[str, Any]:
        target = state.get("target")
        locked = state.get("locked_pose")
        st = api.get_state()
        snap = _apply_state_snapshot(state, st)

        # Prefer last_placed_piece if the engine populated it; else client-tracked pose.
        last = st.get("last_placed_piece")
        actual: Optional[Placement] = None
        if last and state.get("target_piece_type") and last.get("type") == state.get(
            "target_piece_type"
        ):
            # last_placed is only set by /place today; keep for forward-compat
            actual = (last["row"], last["col"], last["rotation"] % 4)
        elif locked is not None:
            actual = (locked[0], locked[1], locked[2] % 4)

        ok = False
        if target is not None and actual is not None:
            ok = (
                actual[0] == target[0]
                and actual[1] == target[1]
                and actual[2] == target[2] % 4
            )
        elif target is not None and st.get("pieces_placed", 0) > state.get(
            "pieces_placed_before", 0
        ):
            # Locked something but we could not recover the pose — treat as miss.
            ok = False

        if ok:
            log.info(
                "validation OK piece=%s intended=%s actual=%s lines=%s",
                state.get("target_piece_type"),
                _fmt_pose(target),
                _fmt_pose(actual),
                snap.get("lines_cleared", 0),
            )
            return {
                **snap,
                "placement_ok": True,
                "llm_assist_once": False,
                "failure_context": None,
            }

        tgt_s = _fmt_pose(target)
        act_s = _fmt_pose(actual) if actual else "unknown/unlocked"
        ctx = (
            f"Previous placement attempt missed. Intended {tgt_s} for piece "
            f"{state.get('target_piece_type')}, but locked at {act_s}. "
            f"Choose a different strong legal placement for the NEW current piece."
        )
        log.warning(
            "validation MISS piece=%s intended=%s actual=%s → "
            "next placement will use LLM assist",
            state.get("target_piece_type"),
            tgt_s,
            act_s,
        )
        return {
            **snap,
            "placement_ok": False,
            "llm_assist_once": True,
            "failure_context": ctx,
        }

    def route_after_state(state: AgentPlayState) -> str:
        if state.get("done"):
            return "end"
        return "plan"

    def route_after_plan(state: AgentPlayState) -> str:
        if state.get("done") or not state.get("target"):
            return "end"
        return "move"

    graph = StateGraph(AgentPlayState)
    graph.add_node("game_state_agent", game_state_agent)
    graph.add_node("final_placement_agent", final_placement_agent)
    graph.add_node("movement_agent", movement_agent)
    graph.add_node("validation_agent", validation_agent)

    graph.set_entry_point("game_state_agent")
    graph.add_conditional_edges(
        "game_state_agent",
        route_after_state,
        {"plan": "final_placement_agent", "end": END},
    )
    graph.add_conditional_edges(
        "final_placement_agent",
        route_after_plan,
        {"move": "movement_agent", "end": END},
    )
    graph.add_edge("movement_agent", "validation_agent")
    graph.add_edge("validation_agent", "game_state_agent")

    return graph.compile()


def _board(board_state: List[List[int]]) -> Board:
    b = Board(width=len(board_state[0]), height=len(board_state))
    b.grid = [r[:] for r in board_state]
    return b


def _piece_pose(cur: Optional[dict]) -> Optional[Placement]:
    if not cur:
        return None
    return (int(cur["row"]), int(cur["col"]), int(cur["rotation"]) % 4)


def _infer_locked_pose(
    before: Optional[Placement],
    move: MoveName,
    board_state: List[List[int]],
    piece_type: PieceType,
) -> Optional[Placement]:
    if before is None:
        return None
    piece = Piece(piece_type, row=before[0], col=before[1], rotation=before[2])
    board = _board(board_state)
    if move == "left":
        piece = piece.move(0, -1)
    elif move == "right":
        piece = piece.move(0, 1)
    elif move == "rotate":
        piece = piece.rotate()
    elif move == "down":
        piece = piece.move(1, 0)
    elif move == "drop":
        while board.is_valid_position(piece.move(1, 0).get_cells()):
            piece = piece.move(1, 0)
    return (piece.row, piece.col, piece.rotation % 4)


def _llm_choose_placement(
    llm: ChatOpenAI,
    board: List[List[int]],
    piece_type: PieceType,
    candidates: List[Placement],
    failure_context: Optional[str],
) -> Tuple[Optional[Placement], Optional[str]]:
    """Return (placement, reason). Placement is None if LLM fails / illegal."""
    cand_lines = [
        f"{i}: row={r}, col={c}, rotation={rot}"
        for i, (r, c, rot) in enumerate(candidates)
    ]
    prompt = (
        "You are helping place a Tetris-like piece. Pick ONE placement from the "
        "legal candidate list only. Reply with JSON fields row, col, rotation, reason.\n\n"
        f"Piece: {piece_type.value}\n"
        f"Board (#=filled, .=empty):\n{board_to_ascii(board)}\n\n"
        f"Legal candidates:\n" + "\n".join(cand_lines) + "\n"
    )
    if failure_context:
        prompt += f"\nContext from prior miss:\n{failure_context}\n"

    log.debug("llm_placement prompt:\n%s", prompt)
    try:
        structured = llm.with_structured_output(ChosenPlacement)
        result: ChosenPlacement = structured.invoke(prompt)
        pick = (result.row, result.col, result.rotation % 4)
        legal = {(r, c, rot % 4) for r, c, rot in candidates}
        if pick in legal:
            return pick, result.reason
        for r, c, rot in candidates:
            if r == result.row and c == result.col and (rot % 4) == (result.rotation % 4):
                return (r, c, rot % 4), result.reason
        log.warning(
            "llm_placement illegal pick %s (not in %d candidates) reason=%r",
            _fmt_pose(pick),
            len(candidates),
            result.reason,
        )
    except Exception as exc:  # noqa: BLE001 — fall back to transformer
        log.error("llm_placement call failed: %s", exc)
    return None, None


def _llm_or_drop_recover(
    api: GameAPI,
    get_llm,
    target: Placement,
    board: List[List[int]],
    piece_type: PieceType,
    budget: int,
) -> Optional[Placement]:
    """Ask LLM for next moves when BFS path failed; finish with drop if possible."""
    pieces_before = api.get_state().get("pieces_placed", 0)
    llm = get_llm()
    log.info(
        "llm_recover start piece=%s target=%s budget=%d",
        piece_type.value,
        _fmt_pose(target),
        budget,
    )

    class NextMove(BaseModel):
        move: Literal["left", "right", "rotate", "down", "drop"]
        done: bool = False

    for step_i in range(max(0, budget)):
        st = api.get_state()
        if st.get("pieces_placed", 0) > pieces_before:
            log.info("llm_recover: piece already locked mid-loop")
            return None
        cur = st.get("current_piece")
        if not cur:
            return None
        pose = _piece_pose(cur)
        if pose and pose[1] == target[1] and pose[2] == target[2] % 4:
            before = pose
            st = api.move("drop")
            if st.get("pieces_placed", 0) > pieces_before:
                locked = _infer_locked_pose(before, "drop", board, piece_type)
                log.info("llm_recover: col/rot matched — drop locked at %s",
                         _fmt_pose(locked))
                return locked
            return None

        prompt = (
            "Navigate the falling piece to the target placement using one move. "
            "Valid moves: left, right, rotate (clockwise), down, drop.\n"
            f"Current pose: row={cur['row']}, col={cur['col']}, rot={cur['rotation']}\n"
            f"Target: row={target[0]}, col={target[1]}, rot={target[2]}\n"
            f"Piece: {piece_type.value}\n"
            f"Board:\n{board_to_ascii(st['board'])}\n"
            "Pick the single best next move. Set done=true only if you just want to hard-drop."
        )
        try:
            nxt: NextMove = llm.with_structured_output(NextMove).invoke(prompt)
            log.info(
                "llm_recover step=%d at=%s → move=%s done=%s",
                step_i,
                _fmt_pose(pose),
                nxt.move,
                nxt.done,
            )
            before = _piece_pose(cur)
            st = api.move(nxt.move)
            if st.get("pieces_placed", 0) > pieces_before:
                locked = _infer_locked_pose(before, nxt.move, board, piece_type)
                log.info("llm_recover locked via %s at %s", nxt.move, _fmt_pose(locked))
                return locked
            if nxt.done:
                before = _piece_pose(st.get("current_piece"))
                st = api.move("drop")
                if st.get("pieces_placed", 0) > pieces_before:
                    locked = _infer_locked_pose(before, "drop", board, piece_type)
                    log.info("llm_recover done+drop locked at %s", _fmt_pose(locked))
                    return locked
                return None
        except Exception as exc:  # noqa: BLE001
            log.error("llm_recover call failed: %s — hard-dropping", exc)
            before = _piece_pose(cur)
            st = api.move("drop")
            if st.get("pieces_placed", 0) > pieces_before:
                return _infer_locked_pose(before, "drop", board, piece_type)
            return None
    log.warning("llm_recover exhausted budget without lock")
    return None


# ---------------------------------------------------------------------------
# Play loop / CLI
# ---------------------------------------------------------------------------

def play(
    model_path: str,
    api_url: str = "http://localhost:8000",
    allowed_pieces: Optional[List[str]] = None,
    max_moves: int = 400,
    openai_model: str = "gpt-4o-mini",
    log_level: str = "INFO",
) -> dict:
    configure_logging(log_level)
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY not set; LLM assist will fail if needed.")

    api = GameAPI(api_url)
    scorer = TransformerScorerPlayer(model_path, api_url=api_url, use_inprocess=False)
    # Scorer only used for choose/score_candidates; keep its game_id unused.

    initial = api.create_game(allowed_pieces=allowed_pieces)
    log.info(
        "game created id=%s api=%s model=%s openai=%s max_moves=%s",
        initial.get("game_id"),
        api_url,
        model_path,
        openai_model,
        max_moves,
    )
    app = build_graph(api, scorer, openai_model=openai_model)

    state: AgentPlayState = {
        "game_id": api.game_id or initial["game_id"],
        "api_url": api_url,
        "board": initial["board"],
        "current_piece": initial.get("current_piece"),
        "status": initial.get("status", "playing"),
        "score": initial.get("score", 0),
        "lines_cleared": initial.get("lines_cleared", 0),
        "pieces_placed": initial.get("pieces_placed", 0),
        "target": None,
        "llm_assist_once": False,
        "failure_context": None,
        "move_count": 0,
        "max_moves": max_moves,
        "openai_model": openai_model,
        "done": False,
    }

    # Recursion limit: each piece ≈ 4 node visits; allow headroom.
    final = app.invoke(state, config={"recursion_limit": max(50, max_moves * 6)})
    return api.get_state() if api.game_id else final


def main():
    ap = argparse.ArgumentParser(
        description="Replay with LangGraph agents (transformer + LLM fallback)"
    )
    ap.add_argument(
        "--model",
        default="./models/transformer_scorer/transformer_scorer_best.keras",
    )
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--allowed-pieces", type=str, nargs="*", default=None)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Agent activity logging (DEBUG includes BFS path + LLM prompts)",
    )
    args = ap.parse_args()

    final = play(
        model_path=args.model,
        api_url=args.api_url,
        allowed_pieces=args.allowed_pieces,
        max_moves=args.max_moves,
        openai_model=args.openai_model,
        log_level=args.log_level,
    )
    log.info(
        "FINAL lines=%s pieces=%s score=%s status=%s",
        final.get("lines_cleared", 0),
        final.get("pieces_placed", 0),
        final.get("score", 0),
        final.get("status"),
    )
    print(f"Lines cleared: {final.get('lines_cleared', 0)}")
    print(f"Pieces placed: {final.get('pieces_placed', 0)}")
    print(f"Score: {final.get('score', 0)}")
    print(f"Status: {final.get('status')}")


if __name__ == "__main__":
    main()
