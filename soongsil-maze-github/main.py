"""Terminal controls and batch comparison. No third-party packages required."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from time import monotonic, sleep

from agents import ALGORITHM_LABELS, PHASE_LABELS
from experiment_log import EndReason, REASON_LABELS, SavedRun
from game import DEFAULT_MAX_ACTIONS, ExperimentGame
from levels import MAPS, MAP_SHORTCUTS

from maze_env import (
    Action,
    AgentMemory,
    EventKind,
    Observation,
    StepResult,
)


KEY_ACTIONS = {
    "w": Action.UP,
    "s": Action.DOWN,
    "a": Action.LEFT,
    "d": Action.RIGHT,
    "up": Action.UP,
    "down": Action.DOWN,
    "left": Action.LEFT,
    "right": Action.RIGHT,
}


ManualGame = ExperimentGame  # Retain the existing import used by manual clients.
AGENT_SHORTCUTS = {"4": "dfs", "5": "bfs", "6": "astar", "m": "manual"}


def saved_note(saved: SavedRun) -> str:
    return f"기록 저장: {saved.text_path.name} (+ JSON)"


def render_frame(
    observation: Observation, memory: AgentMemory, message: str, map_name: str | None = None,
    *, controller: str = "manual", running: bool = False, computation: dict | None = None,
    end_reason: EndReason | None = None, attempts: int = 0, max_actions: int = DEFAULT_MAX_ACTIONS,
) -> str:
    """Render only current observations, never the remembered/full map."""
    visible = {cell.position: cell.tile.value for cell in observation.visible}
    height, width = observation.shape
    lines = [
        "열쇠를 찾아 탈출하기 - " + ALGORITHM_LABELS[controller] + (f" / {map_name}" if map_name else ""),
        "좌표: (행, 열), 0부터 시작 / ? = 현재 시야 밖",
        "",
        "     " + " ".join(f"{col:02}" for col in range(width)),
    ]
    for row in range(height):
        symbols = [
            "@" if (row, col) == observation.position else visible.get((row, col), "?")
            for col in range(width)
        ]
        lines.append(f"{row:02} | " + " ".join(f"{symbol:>2}" for symbol in symbols))

    lines.extend(
        [
            "",
            "@ 나 / # 벽 / . 통로 / S 시작 / E 출구 / K 열쇠 / T 발견한 함정",
            f"위치 {observation.position} | 출구 {observation.exit_position} "
            f"| 열쇠 {'있음' if observation.has_key else '없음'}",
            f"누적 이동 {observation.moves}회 | 함정 {observation.trap_hits}회 "
            f"| 방문한 칸 {len(memory.visited_positions)}개 "
            f"| 기억한 함정 {len(memory.known_traps)}개",
            "방향키 또는 WASD: 이동 / R: 새 실험(기억과 기록 초기화) / Q: 종료",
            "맵 변경: 1 기본15 / 2 우회15 / 3 도전21 / 7 비교41 (현재 실험을 저장한 뒤 전환)",
            "실행 주체: M 수동 / 4 온라인 DFS / 5 BFS / 6 A* 활용형 (변경 시 새 실험)",
            "",
            message,
        ]
    )
    if controller != "manual":
        metrics = computation or {}
        lines.extend([
            f"{'자동 실행 중' if running else '일시정지'} | N/Space/Enter: 한 칸 / P: 자동 실행·정지",
            f"행동 시도 {attempts}/{max_actions} | 확장 노드 {metrics.get('expanded_nodes', 0)}개 "
            f"| 판단 {metrics.get('decision_time_seconds', 0):.6f}초 "
            f"| 경로 검색 {metrics.get('path_planning_seconds', 0):.6f}초",
        ])
    if end_reason is not None and not observation.done:
        lines.append(f"실험 종료: {REASON_LABELS[end_reason]} / R 새 실험, Q 종료")
    if observation.done:
        lines.append("탈출 성공! R로 새 실험을 시작하거나 Q로 종료하세요.")
    return "\n".join(lines)


def describe_result(result: StepResult) -> str:
    messages = []
    for event in result.events:
        if event.kind == EventKind.WALL_BLOCKED:
            messages.append("벽에 막혔습니다. 이동 횟수는 증가하지 않습니다.")
        elif event.kind == EventKind.EXIT_LOCKED:
            messages.append("출구가 잠겨 있습니다. 열쇠가 있어야 들어갈 수 있습니다.")
        elif event.kind == EventKind.KEY_PICKED_UP:
            messages.append("열쇠를 얻었습니다. 출구를 찾아가세요.")
        elif event.kind == EventKind.WAITED:
            messages.append("한 턴 기다렸습니다.")
        elif event.kind == EventKind.TRAP_TRIGGERED:
            messages.append(
                f"함정 {event.position}! 시작점으로 돌아왔습니다. "
                "지도 기억과 누적 기록은 유지됩니다."
            )
            if result.observation.has_key:
                messages.append("열쇠도 계속 보유합니다.")
        elif event.kind == EventKind.GHOST_COLLISION:
            messages.append("유령에 닿아 시작점으로 돌아왔습니다.")
            if result.observation.has_key:
                messages.append("열쇠는 계속 보유합니다.")
        elif event.kind == EventKind.KEY_RESPAWNED:
            messages.append("열쇠를 잃었습니다. 원래 위치에 다시 생성되었습니다.")
        elif event.kind == EventKind.ESCAPED:
            messages.append(
                f"탈출했습니다. 총 이동 {result.observation.moves}회, "
                f"함정 {result.observation.trap_hits}회."
            )
        elif event.kind == EventKind.ALREADY_FINISHED:
            messages.append("이미 탈출했습니다. R로 새 실험을 시작할 수 있습니다.")
    return " ".join(messages) or "이동했습니다."


def read_command(immediate: bool, *, eof_command: str = "q") -> str:
    """Windows console: one key. Redirected/other input: one command per line."""
    if not immediate:
        try:
            return input("행동> ").strip().lower()
        except EOFError:
            return eof_command

    import msvcrt

    key = msvcrt.getwch()
    if key == "\x03":
        raise KeyboardInterrupt
    if key in ("\x00", "\xe0"):
        arrow = msvcrt.getwch()
        return {"H": "w", "P": "s", "K": "a", "M": "d"}.get(arrow, "")
    if key in ("\r", "\n"):
        return ""
    return key.lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="부분 관측 미로: 수동 조작 및 DFS/BFS/A* 비교")
    parser.add_argument(
        "--line-input", action="store_true", help="WASD + Enter 입력 사용"
    )
    parser.add_argument(
        "--no-clear", action="store_true", help="이전 화면을 지우지 않고 출력"
    )
    parser.add_argument(
        "--log-dir", type=Path, help="기록 저장 폴더 (기본: 프로그램 폴더의 logs)"
    )
    parser.add_argument(
        "--map", dest="map_id", choices=tuple(MAPS), default="basic15",
        help="시작할 고정 맵 선택 (기본: basic15)",
    )
    parser.add_argument("--list-maps", action="store_true", help="맵 목록만 출력하고 종료")
    parser.add_argument("--agent", choices=tuple(ALGORITHM_LABELS), default="manual", help="실행 주체 (기본: manual)")
    parser.add_argument("--auto", action="store_true", help="선택한 에이전트를 자동 실행 상태로 시작")
    parser.add_argument("--headless", action="store_true", help="화면/입력 대기 없이 에이전트 한 실험을 실행하고 종료")
    parser.add_argument("--no-traps", action="store_true", help="고정 함정을 끄고 실행")
    parser.add_argument("--compare", action="store_true", help="전체 맵 × 세 에이전트 비교 및 동일 지도 BFS/A* 경로 비교 CSV 저장")
    parser.add_argument("--key-placements", action="store_true", help="--compare에서 기본 맵의 열쇠 배치 3개 비교")
    parser.add_argument("--repeat", type=int, default=1, help="비교 반복 횟수; 매회 기억 초기화, 시간 재측정용 (기본: 1)")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS, help="자동 실험 행동 시도 한도, 벽 충돌 포함 (기본: 5000)")
    parser.add_argument("--delay", type=float, default=0.12, help="자동 화면 실행의 이동 간격, 초 (기본: 0.12)")
    args = parser.parse_args(argv)
    if args.key_placements and not args.compare:
        parser.error("--key-placements는 --compare와 함께 사용하세요.")
    if args.max_actions < 1 or args.repeat < 1:
        parser.error("--max-actions와 --repeat는 1 이상이어야 합니다.")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay는 0 이상의 유한한 수여야 합니다.")

    if args.list_maps:
        for shortcut, map_id in MAP_SHORTCUTS.items():
            definition = MAPS[map_id]
            print(f"{shortcut}: {definition.map_id} / {definition.name} / {definition.description}")
        return 0
    if args.compare:
        from comparison import run_comparison

        try:
            return run_comparison(
                args.log_dir, repeats=args.repeat, max_actions=args.max_actions, key_placements=args.key_placements, traps_enabled=not args.no_traps,
            )
        except OSError as error:
            print(f"비교 기록 저장 실패: {error}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n비교를 중단했습니다. 완료된 실험 기록은 유지됩니다.")
            return 130
    if (args.auto or args.headless) and args.agent == "manual":
        parser.error("--auto/--headless는 --agent dfs, bfs, astar 중 하나와 함께 사용하세요.")

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    immediate = os.name == "nt" and interactive and not args.line_input
    clear = interactive and not args.no_clear

    definition = MAPS[args.map_id]
    game = ManualGame(
        definition.layout, log_dir=args.log_dir,
        map_id=definition.map_id, map_name=definition.name,
        controller=args.agent, max_actions=args.max_actions, traps_enabled=not args.no_traps,
    )
    if args.headless:
        try:
            try:
                while not game.finished:
                    game.automatic_step()
            except KeyboardInterrupt:
                game.finish(EndReason.INTERRUPTED)
        except OSError as error:
            print(f"기록 저장 실패: {error}", file=sys.stderr)
            return 1
        print(f"{game.map_id} / {ALGORITHM_LABELS[game.controller]} / {REASON_LABELS[game.end_reason]}")
        print(f"이동 {game.observation.moves}회, 함정 {game.observation.trap_hits}회, "
              f"행동 시도 {game.recorder.action_attempts}회")
        print(f"확장 노드 {game.recorder.computation['expanded_nodes']}개, "
              f"판단 계산 {game.recorder.computation['decision_time_seconds']:.6f}초")
        if game.last_saved:
            print(f"TXT: {game.last_saved.text_path}\nJSON: {game.last_saved.json_path}")
        return 0 if game.observation.done else 2
    message = "열쇠 위치는 모릅니다. 주변을 탐색하세요. 함정은 밟기 전에는 보이지 않습니다."
    if not immediate and game.agent is None:
        message += " WASD를 입력하고 Enter를 누르세요."
    if game.agent is not None:
        message = "N 또는 Enter로 한 칸, P로 자동 실행합니다. " + message
    running = args.auto
    next_step = monotonic()
    redraw = True

    def agent_step_message() -> str:
        result = game.automatic_step()
        note = describe_result(result) if result is not None else REASON_LABELS[game.end_reason]
        if game.last_decision is not None:
            note += f"\n판단: {PHASE_LABELS[game.last_decision.phase]}"
            if game.last_decision.target is not None:
                note += f" / 목표 {game.last_decision.target}"
        if game.last_saved:
            note += "\n" + saved_note(game.last_saved)
        return note

    try:
        while True:
            if redraw:
                if clear:
                    os.system("cls" if os.name == "nt" else "clear")
                print(render_frame(
                    game.observation, game.memory, message, game.map_name,
                    controller=game.controller, running=running, computation=game.recorder.computation,
                    end_reason=game.end_reason, attempts=game.recorder.action_attempts, max_actions=game.max_actions,
                ), flush=True)
                if running and not immediate:
                    print("자동 실행 중에는 Ctrl+C로 중단할 수 있습니다.", flush=True)
                redraw = False
            command = None
            if running and not game.finished:
                if immediate:
                    import msvcrt

                    if msvcrt.kbhit():
                        command = read_command(immediate)
                if command is None:
                    if monotonic() < next_step:
                        sleep(max(0.0, min(0.02, next_step - monotonic())))
                        continue
                    try:
                        message = agent_step_message()
                    except OSError as error:
                        message = f"기록 저장 실패: {error}\n현재 실험을 유지합니다. Q로 저장을 다시 시도하세요."
                        running = False
                    next_step = monotonic() + args.delay
                    if game.finished:
                        running = False
                    redraw = True
                    continue
            if command is None:
                command = read_command(immediate, eof_command="__eof__")
            redraw = True
            try:
                if command in {"q", "__eof__"}:
                    game.finish(EndReason.QUIT)
                    break
                if command == "r":
                    saved = game.restart()
                    running = False
                    message = saved_note(saved) + "\n새 실험입니다. 지도 기억, 방문 기록, 이동 및 함정 횟수를 초기화했습니다."
                elif command in MAP_SHORTCUTS:
                    selected = MAPS[MAP_SHORTCUTS[command]]
                    if selected.map_id == game.map_id:
                        message = "현재 선택한 맵입니다. R로 새 실험을 시작할 수 있습니다."
                    else:
                        saved = game.change_map(selected.layout, selected.map_id, selected.name)
                        running = False
                        message = saved_note(saved) + f"\n{selected.name} 시작: {selected.description}"
                elif command in AGENT_SHORTCUTS:
                    selected_controller = AGENT_SHORTCUTS[command]
                    if selected_controller == game.controller:
                        message = "현재 실행 주체입니다. R로 새 실험을 시작할 수 있습니다."
                    else:
                        saved = game.change_controller(selected_controller)
                        running = False
                        controls = "WASD 또는 방향키로 이동" if game.agent is None else "N 한 칸 / P 자동 실행"
                        message = saved_note(saved) + f"\n{ALGORITHM_LABELS[selected_controller]} 새 실험. {controls}"
                elif command == "p" and game.agent is not None:
                    if game.finished:
                        message = "실험이 끝났습니다. R로 다시 시작하세요."
                    else:
                        running = not running
                        next_step = monotonic()
                        message = "자동 실행합니다." if running else "일시정지했습니다. N으로 한 칸 실행할 수 있습니다."
                elif command in {"n", " ", ""} and game.agent is not None:
                    running = False
                    message = "실험이 끝났습니다. R로 다시 시작하세요." if game.finished else agent_step_message()
                elif command in KEY_ACTIONS:
                    if game.agent is not None:
                        message = "에이전트 모드입니다. N 한 칸 / P 자동 실행 / M 수동 전환"
                    else:
                        result = game.move(KEY_ACTIONS[command])
                        message = describe_result(result)
                        if result.done and game.last_saved:
                            message += "\n" + saved_note(game.last_saved)
                else:
                    message = "WASD 수동 이동 / 1·2·3·7 맵 / 4·5·6 에이전트 / N 한 칸 / P 자동 / R 새 실험 / Q 종료"
            except OSError as error:
                if command == "__eof__":
                    print(f"입력이 끝났으며 기록 저장에 실패했습니다: {error}", file=sys.stderr)
                    return 1
                message = f"기록 저장 실패: {error}\n현재 실험을 유지합니다. R/Q로 저장을 다시 시도할 수 있습니다."
                running = False
    except KeyboardInterrupt:
        print("\n실험을 종료합니다.")
        try:
            game.finish(EndReason.INTERRUPTED)
        except OSError as error:
            print(f"기록을 저장하지 못했습니다: {error}", file=sys.stderr)
            return 1
    observation = game.observation
    print(f"종료 기록: 이동 {observation.moves}회, 함정 {observation.trap_hits}회.")
    if game.last_saved:
        print(f"TXT: {game.last_saved.text_path}\nJSON: {game.last_saved.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

