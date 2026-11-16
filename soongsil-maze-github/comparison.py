"""Fresh-run agent comparisons and paired routing on identical public snapshots."""

from __future__ import annotations

import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from uuid import uuid4

from agents import RoutingTask, astar_route, bfs_route
from experiment_log import EndReason, KST
from game import DEFAULT_MAX_ACTIONS, ExperimentGame
from levels import BASIC_KEY_PLACEMENTS, MAPS


SUMMARY_FIELDS = (
    "repeat", "map_id", "algorithm", "traps_enabled", "run_id", "status", "end_reason",
    "moves", "action_attempts", "trap_hits", "known_cells", "visited_cells",
    "decision_calls", "decision_time_seconds", "path_planning_seconds",
    "expanded_nodes", "route_expanded_nodes", "exploration_expanded_nodes",
    "search_calls", "peak_frontier", "max_action_attempts", "json_file",
)
ROUTE_FIELDS = (
    "map_id", "task", "phase", "snapshot_sha256", "known_cells",
    "start_row", "start_col", "has_key", "goal_row", "goal_col",
    "bfs_path_length", "astar_path_length", "same_path_length",
    "bfs_expanded_nodes", "astar_expanded_nodes", "bfs_peak_frontier", "astar_peak_frontier",
    "timing_repeats", "bfs_seconds", "astar_seconds",
)


def _write_csv(path: Path, fields, rows) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def benchmark_task(map_id: str, index: int, task: RoutingTask, *, repeats: int = 1) -> dict:
    """Both planners receive the exact same immutable graph, state and goal."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    timings = {"bfs": [], "astar": []}
    results = {}
    for repetition in range(repeats):
        # Alternate execution order so one planner is not always measured first.
        order = ("bfs", "astar") if (index + repetition) % 2 else ("astar", "bfs")
        for algorithm in order:
            route = bfs_route(task.knowledge, task.start, goal=task.goal) if algorithm == "bfs" else astar_route(task.knowledge, task.start, task.goal)
            results[algorithm] = route
            timings[algorithm].append(route.seconds)
    bfs, astar = results["bfs"], results["astar"]
    bfs_length = None if bfs.actions is None else len(bfs.actions)
    astar_length = None if astar.actions is None else len(astar.actions)
    if bfs_length != astar_length:
        raise RuntimeError("BFS/A* disagree on reachability or shortest known path length.")
    public_graph = {
        "shape": task.knowledge.shape,
        "tiles": [(r, c, tile.value) for (r, c), tile in sorted(task.knowledge.tiles.items())],
        "traps": sorted(task.knowledge.traps),
    }
    fingerprint = sha256(json.dumps(public_graph, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "map_id": map_id, "task": index, "phase": task.phase,
        "snapshot_sha256": fingerprint, "known_cells": len(task.knowledge.tiles),
        "start_row": task.start[0], "start_col": task.start[1], "has_key": task.start[2],
        "goal_row": task.goal[0], "goal_col": task.goal[1],
        "bfs_path_length": bfs_length, "astar_path_length": astar_length, "same_path_length": True,
        "bfs_expanded_nodes": bfs.expanded_nodes, "astar_expanded_nodes": astar.expanded_nodes,
        "bfs_peak_frontier": bfs.peak_frontier, "astar_peak_frontier": astar.peak_frontier,
        "timing_repeats": repeats,
        "bfs_seconds": mean(timings["bfs"]), "astar_seconds": mean(timings["astar"]),
    }


def run_comparison(
    directory: Path | str | None = None, *, repeats: int = 1,
    max_actions: int = DEFAULT_MAX_ACTIONS, key_placements: bool = False, traps_enabled: bool = True,
) -> int:
    if repeats < 1 or max_actions < 1:
        raise ValueError("repeats and max_actions must be positive.")
    default_log_dir = Path(__file__).resolve().parent / "logs"
    if key_placements:
        default_log_dir /= "key-placement-comparison"
    directory = Path(directory) if directory is not None else default_log_dir
    directory.mkdir(parents=True, exist_ok=True)
    definitions = BASIC_KEY_PLACEMENTS if key_placements else tuple(MAPS.values())
    map_width = max(10, max(len(definition.map_id) for definition in definitions))
    rows, tasks = [], []
    interrupted = False
    cases = "기본 15×15 열쇠 배치 3개" if key_placements else f"고정 맵 {len(MAPS)}개"
    print(f"{cases} × DFS/BFS/A*: 매 실험 지도 기억과 기록을 초기화합니다.")
    print("각 배치는 세 에이전트에 똑같이 적용합니다. 반복은 계산 시간 재측정용입니다.")
    header = f"{'맵':<{map_width}} 알고리즘  종료             이동  함정  확장노드    판단(s)"
    print(header + ("  열쇠까지" if key_placements else ""))
    for repetition in range(1, repeats + 1):
        for definition in definitions:
            map_id = definition.map_id
            for algorithm in ("dfs", "bfs", "astar"):
                game = ExperimentGame(
                    definition.layout, map_id=map_id, map_name=definition.name,
                    log_dir=directory, controller=algorithm, max_actions=max_actions,
                    capture_tasks=algorithm == "bfs" and repetition == 1, traps_enabled=traps_enabled,
                )
                try:
                    while not game.finished:
                        game.automatic_step()
                except KeyboardInterrupt:
                    game.finish(EndReason.INTERRUPTED)
                    interrupted = True
                if game.agent.capture_tasks:
                    tasks.extend((map_id, index, task) for index, task in enumerate(game.agent.route_tasks, 1))
                status = {
                    EndReason.ESCAPED: "success", EndReason.ACTION_LIMIT: "limit_reached",
                    EndReason.EXHAUSTED: "exhausted",
                }.get(game.end_reason, "aborted")
                row = {
                    "repeat": repetition, "map_id": map_id, "algorithm": algorithm, "traps_enabled": traps_enabled,
                    "run_id": game.recorder.run_id, "status": status, "end_reason": game.end_reason.value,
                    "moves": game.observation.moves, "action_attempts": game.recorder.action_attempts,
                    "trap_hits": game.observation.trap_hits, "known_cells": len(game.memory.known_tiles),
                    "visited_cells": len(game.memory.visited_positions), **game.recorder.computation,
                    "max_action_attempts": max_actions, "json_file": game.last_saved.json_path.name,
                }
                key_column = ""
                if key_placements:
                    moves_to_key = game.recorder.moves_to_first_key
                    row["moves_to_key"] = moves_to_key
                    value = str(moves_to_key) if moves_to_key is not None else "—"
                    key_column = f" {value:>8}"
                rows.append(row)
                print(f"{map_id:<{map_width}} {algorithm:<9} {status:<16} {row['moves']:>4} {row['trap_hits']:>4} "
                      f"{row['expanded_nodes']:>8} {row['decision_time_seconds']:>10.6f}{key_column}", flush=True)
                if interrupted:
                    break
            if interrupted:
                break
        if interrupted:
            break

    prefix = "comparison_keys" if key_placements else "comparison"
    stem = f"{prefix}_{datetime.now(KST):%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"
    summary_path = directory / f"{stem}.csv"
    fields = (*SUMMARY_FIELDS, "moves_to_key") if key_placements else SUMMARY_FIELDS
    _write_csv(summary_path, fields, rows)
    print(f"\n실험 요약 CSV: {summary_path}")
    if not interrupted:
        route_rows = [benchmark_task(map_id, index, task, repeats=repeats) for map_id, index, task in tasks]
        route_path = directory / f"{stem}_routes.csv"
        _write_csv(route_path, ROUTE_FIELDS, route_rows)
        print(f"동일 내부 지도 BFS/A* 경로 비교 CSV: {route_path}")
        print(f"경로 비교 시간은 과제별 {repeats}회 평균입니다.")
        if route_rows:
            print(f"동일 조건 경로 과제 {len(route_rows)}건: 도달 가능 여부와 최단 경로 길이 모두 일치")
            print(f"평균 확장 노드: BFS {mean(row['bfs_expanded_nodes'] for row in route_rows):.2f}, "
                  f"A* {mean(row['astar_expanded_nodes'] for row in route_rows):.2f}")
        print("A*형도 미지 영역 탐색은 BFS형과 같습니다. 전체 이동과 경로 계산량을 구분해 비교하세요.")
    print("개별 실험 JSON/TXT도 같은 폴더에 저장했습니다.")
    return 130 if interrupted else (0 if all(row["status"] == "success" for row in rows) else 2)
