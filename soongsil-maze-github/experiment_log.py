"""End-of-experiment JSON/TXT records, built only from public observations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic
from uuid import uuid4
from dataclasses import dataclass

from agents import PHASE_LABELS
from maze_env import Action, AgentMemory, EventKind, MazeLayout, Observation, Position, StepResult


KST = timezone(timedelta(hours=9), name="KST")
SCHEMA_VERSION = 1


class EndReason(str, Enum):
    ESCAPED = "escaped"
    QUIT = "quit"
    RESET = "reset"
    MAP_CHANGED = "map_changed"
    CONTROLLER_CHANGED = "controller_changed"
    GHOST_MODE_CHANGED = "ghost_mode_changed"
    TRAP_MODE_CHANGED = "trap_mode_changed"
    INTERRUPTED = "interrupted"
    ACTION_LIMIT = "action_limit"
    EXHAUSTED = "exhausted"


REASON_LABELS = {
    EndReason.ESCAPED: "탈출 성공",
    EndReason.QUIT: "사용자 종료 또는 입력 종료",
    EndReason.RESET: "새 실험 시작(R)",
    EndReason.MAP_CHANGED: "맵 변경",
    EndReason.CONTROLLER_CHANGED: "실행 주체 변경",
    EndReason.GHOST_MODE_CHANGED: "순찰 유령 설정 변경",
    EndReason.TRAP_MODE_CHANGED: "고정 함정 설정 변경",
    EndReason.INTERRUPTED: "사용자 중단(Ctrl+C)",
    EndReason.ACTION_LIMIT: "행동 시도 한도 초과",
    EndReason.EXHAUSTED: "현재 지식에서 탐색할 경로 없음",
}
STATUS_LABELS = {
    "success": "성공", "aborted": "중단 (실패 판정 아님)",
    "limit_reached": "한도 초과 (탈출 불가능 판정 아님)",
    "exhausted": "탐색 종료 (현재 지식에서 진행할 경로 없음)",
}
ACTION_LABELS = {"up": "위", "down": "아래", "left": "왼쪽", "right": "오른쪽", "wait": "대기"}
EVENT_LABELS = {
    "moved": "이동",
    "waited": "대기",
    "wall_blocked": "벽 충돌",
    "exit_locked": "잠긴 출구",
    "key_picked_up": "열쇠 획득",
    "trap_triggered": "함정 발동",
    "ghost_collision": "유령 접촉",
    "key_respawned": "열쇠 소실/원위치 재생성",
    "escaped": "탈출 성공",
}


@dataclass(frozen=True)
class SavedRun:
    json_path: Path
    text_path: Path
    run_id: str
    reason: EndReason
    success: bool


def observation_data(observation: Observation) -> dict:
    return {
        "shape": list(observation.shape),
        "position": list(observation.position),
        "start_position": list(observation.start_position),
        "exit_position": list(observation.exit_position),
        "has_key": observation.has_key,
        "moves": observation.moves,
        "trap_hits": observation.trap_hits,
        "turns": observation.turns,
        "ghost_hits": observation.ghost_hits,
        "ghost_position": list(observation.ghost_position) if observation.ghost_position is not None else None,
        "done": observation.done,
        "visible": [
            {"position": list(cell.position), "tile": cell.tile.value}
            for cell in observation.visible
        ],
    }


def memory_data(memory: AgentMemory, shape: tuple[int, int]) -> dict:
    height, width = shape
    grid = [["?" for _ in range(width)] for _ in range(height)]
    for (row, col), tile in memory.known_tiles.items():
        grid[row][col] = tile.value
    return {
        "known_tiles": [
            {"position": list(position), "tile": tile.value}
            for position, tile in sorted(memory.known_tiles.items())
        ],
        "known_grid": ["".join(row) for row in grid],
        "visited_positions": [list(position) for position in sorted(memory.visited_positions)],
        "known_traps": [list(position) for position in sorted(memory.known_traps)],
        "travel_history": [list(position) for position in memory.travel_history],
    }


def text_report(data: dict) -> str:
    summary = data["summary"]
    final = data["final_observation"]
    map_info = data["map"]
    reason = EndReason(data["termination"]["reason"])
    lines = [
        "열쇠 미로 실험 기록",
        "=" * 60,
        f"실험 ID: {data['run_id']}",
        f"맵: {map_info['name']} ({map_info['id']}) / 크기 {tuple(map_info['shape'])}",
        f"맵 SHA-256: {map_info['layout_sha256']}",
        f"실행 주체: {data['controller']}",
        f"시작 시각: {data['started_at']}",
        f"종료 시각: {data['ended_at']}",
        f"종료 사유: {REASON_LABELS[reason]}",
        f"결과: {STATUS_LABELS[data['termination']['status']]}",
        f"실제 경과 시간: {data['elapsed_wall_seconds']:.3f}초",
        "경과 시간에는 입력 대기, 화면 출력, 실행 간격이 포함됩니다. 알고리즘 계산 시간이 아닙니다.",
        f"판단 계산 시간: {data['computation']['decision_time_seconds']:.9f}초 (입력/화면/저장 제외)",
        f"경로 검색 시간: {data['computation']['path_planning_seconds']:.9f}초 (판단 시간의 일부)",
        f"누적 확장 노드: {data['computation']['expanded_nodes']}개 / 검색 호출: {data['computation']['search_calls']}회",
        f"알려진 목표 경로 확장: {data['computation']['route_expanded_nodes']}개 / 탐색 확장: {data['computation']['exploration_expanded_nodes']}개",
        f"최대 검색 대기열/DFS 경로 크기: {data['computation']['peak_frontier']}개",
        "확장 노드는 후속 이동 후보를 검사한 상태 수이며, 재계획에서 반복 검사하면 다시 집계합니다.",
        "",
        "[실행 설정]",
        f"고정 함정: {'켜짐' if data['settings'].get('traps_enabled', True) else '꺼짐'}",
        json.dumps(data['settings'], ensure_ascii=False, sort_keys=True),
        "",
        "[누적 기록]",
        f"행동 시도: {summary['action_attempts']}회 / 실제 이동: {summary['moves']}회",
        f"벽 충돌: {summary['wall_collisions']}회 / 잠긴 출구 진입 시도: {summary['locked_exit_attempts']}회",
        f"함정 발동: {summary['trap_hits']}회 (첫 발견 {summary['first_trap_discoveries']}회, 반복 진입 {summary['repeated_trap_entries']}회)",
        f"열쇠 획득: {summary['key_pickups']}회 / 열쇠 소실: {summary['key_losses']}회",
        f"방문한 칸: {summary['visited_cells']}개 / 기억한 칸: {summary['known_cells']}개",
        f"최종 위치: {tuple(final['position'])} / 열쇠: {'있음' if final['has_key'] else '없음'}",
    ]
    if "moves_to_key" in summary:
        if summary["moves_to_key"] is None:
            lines.append("열쇠 미획득")
        else:
            lines.append(f"열쇠 획득까지 {summary['moves_to_key']}회 / 획득 후 {summary['moves_after_key']}회")
    if data["settings"].get("ghost_enabled"):
        lines.append(f"턴 {summary['turns']}회 / 대기 {summary['waits']}회 / 유령 접촉 {summary['ghost_hits']}회")
    lines.extend([
        "",
        "[종료 시점의 내부 지도]",
        "아래는 관측/이벤트로 알게 된 지도입니다. 미관측 실제 맵은 포함하지 않습니다.",
        "@ 최종 위치 / ? 미확인 / # 벽 / . 통로 / S 시작 / E 출구 / K 열쇠 / T 발견한 함정",
    ])
    grid = data["final_memory"]["known_grid"]
    lines.append("     " + " ".join(f"{col:02}" for col in range(map_info["shape"][1])))
    for row, row_text in enumerate(grid):
        symbols = ["@" if [row, col] == final["position"] else char for col, char in enumerate(row_text)]
        lines.append(f"{row:02} | " + " ".join(f"{symbol:>2}" for symbol in symbols))
    lines.extend([
        "",
        "[이동 경로]",
        "함정·유령 접촉과 시작점 자동 귀환이 기록됩니다. 경로 항목 수와 이동 횟수는 다를 수 있습니다.",
        " -> ".join(str(tuple(position)) for position in data["final_memory"]["travel_history"]),
        "",
        "[행동별 기록]",
    ])
    if not data["steps"]:
        lines.append("이동 명령 없이 종료한 실험입니다.")
    for step in data["steps"]:
        before, after = step["before"], step["after"]
        events = ", ".join(
            f"{EVENT_LABELS.get(event['kind'], event['kind'])} {tuple(event['position'])}"
            for event in step["events"]
        )
        trap_note = {
            "first_discovery": " [함정 첫 발견]",
            "repeat": " [알고 있던 함정 재진입]",
        }.get(step["trap_encounter"], "")
        lines.append(
            f"{step['index']:04} {ACTION_LABELS[step['action']]} "
            f"{tuple(before['position'])} -> {tuple(after['position'])} "
            f"| 열쇠 {before['has_key']} -> {after['has_key']} "
            f"| 누적 이동 {after['moves']} | {events}{trap_note}"
        )
        decision = step.get("decision")
        if decision is not None:
            target = tuple(decision["target"]) if decision["target"] is not None else "없음"
            lines.append(
                f"     판단: {PHASE_LABELS.get(decision['phase'], decision['phase'])} / 목표 {target} "
                f"/ 확장 {decision['expanded_nodes']}개 / 판단 계산 {decision['decision_time_seconds']:.9f}초"
            )
    return "\n".join(lines) + "\n"


class RunRecorder:
    """One independent experiment. Finish is idempotent after a successful save.

    The real layout is used only to compute a reproducibility fingerprint.
    No undiscovered key/trap coordinates or ground-truth map are written.
    """

    def __init__(
        self, layout: MazeLayout, map_id: str, map_name: str,
        initial: Observation, memory: AgentMemory, controller: str = "manual",
        settings: dict | None = None,
    ) -> None:
        self.run_id = uuid4().hex
        self.started_at = datetime.now(KST)
        self._started_clock = monotonic()
        self._map = {
            "id": map_id,
            "name": map_name,
            "shape": list(initial.shape),
            "start_position": list(initial.start_position),
            "exit_position": list(initial.exit_position),
            "layout_sha256": sha256("\n".join(layout.rows).encode("utf-8")).hexdigest(),
        }
        self._controller = controller
        self._settings = dict(settings or {})
        self._computation = {
            "decision_calls": 0, "decision_time_seconds": 0.0,
            "path_planning_seconds": 0.0, "expanded_nodes": 0,
            "route_expanded_nodes": 0, "exploration_expanded_nodes": 0,
            "search_calls": 0, "peak_frontier": 0,
        }
        self._last_decision: dict | None = None
        self._initial = observation_data(initial)
        self._last_observation = initial
        self._known_traps = set(memory.known_traps)
        self._steps: list[dict] = []
        self._moves_to_first_key: int | None = None
        self._saved: SavedRun | None = None

    @property
    def action_attempts(self) -> int:
        return len(self._steps)

    @property
    def travel_edges(self) -> tuple[tuple[Position, Position], ...]:
        """Actual movement segments recorded before any automatic return."""
        return tuple(
            (tuple(step["before"]["position"]), tuple(event["position"]))
            for step in self._steps for event in step["events"]
            if event["kind"] == EventKind.MOVED.value
        )

    @property
    def moves_to_first_key(self) -> int | None:
        """Actual moves through the first key pickup, including that move."""
        return self._moves_to_first_key

    @property
    def moves_after_first_key(self) -> int | None:
        if self._moves_to_first_key is None:
            return None
        return self._last_observation.moves - self._moves_to_first_key

    @property
    def computation(self) -> dict:
        return dict(self._computation)

    def record_decision(self, details: dict) -> None:
        if self._saved is not None or self._last_observation.done:
            raise RuntimeError("A finished experiment cannot accept new decisions.")
        self._last_decision = dict(details)
        self._computation["decision_calls"] += 1
        for key in self._computation:
            if key == "peak_frontier":
                self._computation[key] = max(self._computation[key], details.get(key, 0))
            elif key != "decision_calls":
                self._computation[key] += details.get(key, 0)

    def record(self, action: Action | str, result: StepResult, *, decision: dict | None = None) -> None:
        if self._saved is not None or self._last_observation.done:
            raise RuntimeError("A finished experiment cannot accept new actions.")
        action = Action(action)
        trap_encounter = None
        for event in result.events:
            if event.kind == EventKind.TRAP_TRIGGERED:
                trap_encounter = "repeat" if event.position in self._known_traps else "first_discovery"
                self._known_traps.add(event.position)
            elif event.kind == EventKind.KEY_PICKED_UP and self._moves_to_first_key is None:
                self._moves_to_first_key = result.observation.moves
        self._steps.append({
            "index": len(self._steps) + 1,
            "elapsed_wall_seconds": round(monotonic() - self._started_clock, 6),
            "action": action.value,
            "before": observation_data(self._last_observation),
            "after": observation_data(result.observation),
            "events": [
                {"kind": event.kind.value, "position": list(event.position)}
                for event in result.events
            ],
            "trap_encounter": trap_encounter,
            "decision": dict(decision) if decision is not None else None,
        })
        self._last_observation = result.observation

    def finish(
        self, reason: EndReason | str, observation: Observation,
        memory: AgentMemory, directory: Path | str,
    ) -> SavedRun:
        if self._saved is not None:
            return self._saved
        reason = EndReason(reason)
        if observation != self._last_observation:
            raise ValueError("The final observation must match the last recorded action.")
        if observation.done:
            reason = EndReason.ESCAPED
        elif reason == EndReason.ESCAPED:
            raise ValueError("An unfinished experiment cannot be marked as escaped.")
        status = {
            EndReason.ACTION_LIMIT: "limit_reached", EndReason.EXHAUSTED: "exhausted",
        }.get(reason, "success" if observation.done else "aborted")
        ended_at = datetime.now(KST)
        event_kinds = [event["kind"] for step in self._steps for event in step["events"]]
        summary = {
            "success": observation.done,
            "action_attempts": len(self._steps),
            "moves": observation.moves,
            "moves_to_key": self.moves_to_first_key,
            "moves_after_key": self.moves_after_first_key,
            "trap_hits": observation.trap_hits,
            "turns": observation.turns,
            "waits": event_kinds.count(EventKind.WAITED.value),
            "ghost_hits": observation.ghost_hits,
            "first_trap_discoveries": sum(step["trap_encounter"] == "first_discovery" for step in self._steps),
            "repeated_trap_entries": sum(step["trap_encounter"] == "repeat" for step in self._steps),
            "wall_collisions": event_kinds.count(EventKind.WALL_BLOCKED.value),
            "locked_exit_attempts": event_kinds.count(EventKind.EXIT_LOCKED.value),
            "key_pickups": event_kinds.count(EventKind.KEY_PICKED_UP.value),
            "key_losses": event_kinds.count(EventKind.KEY_RESPAWNED.value),
            "visited_cells": len(memory.visited_positions),
            "known_cells": len(memory.known_tiles),
            "known_trap_cells": len(memory.known_traps),
        }
        data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "map": self._map,
            "controller": self._controller,
            "settings": self._settings,
            "computation": self.computation,
            "last_decision": self._last_decision,
            "started_at": self.started_at.isoformat(timespec="milliseconds"),
            "ended_at": ended_at.isoformat(timespec="milliseconds"),
            "elapsed_wall_seconds": round(monotonic() - self._started_clock, 6),
            "timing_note": "Wall time includes manual input and rendering; this is not algorithm CPU time.",
            "termination": {
                "reason": reason.value,
                "status": status,
            },
            "summary": summary,
            "initial_observation": self._initial,
            "final_observation": observation_data(observation),
            "final_memory": memory_data(memory, observation.shape),
            "steps": self._steps,
        }
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(self._map["id"]))[:48] or "maze"
        stem = f"{self.started_at:%Y%m%d_%H%M%S_%f}_{safe_id}_{self.run_id[:8]}"
        json_path, text_path = directory / f"{stem}.json", directory / f"{stem}.txt"
        # Stage both complete files first. JSON, the structured record, is committed last.
        staged: list[tuple[Path, Path]] = []
        try:
            for destination, content in (
                (text_path, text_report(data)),
                (json_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n"),
            ):
                temporary_path = directory / f".{stem}_{uuid4().hex[:12]}.tmp"
                staged.append((temporary_path, destination))
                with temporary_path.open("x", encoding="utf-8", newline="\n") as temporary:
                    temporary.write(content)
            for temporary, destination in staged:
                temporary.replace(destination)
        finally:
            for temporary, _ in staged:
                if temporary.parent == directory:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass  # Preserve the original save error if cleanup is also denied.
        # Failed saves leave the recorder open so R/Q can retry without losing state.
        self._saved = SavedRun(json_path, text_path, self.run_id, reason, observation.done)
        return self._saved
