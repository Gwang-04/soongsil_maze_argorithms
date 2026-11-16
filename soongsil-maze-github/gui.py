"""Understated Soongsil-themed Tk desktop UI. Run: python gui.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from agents import Decision
from experiment_log import ACTION_LABELS, EndReason, REASON_LABELS
from game import DEFAULT_MAX_ACTIONS, ExperimentGame
from levels import HARD21_GHOST_PATROL, MAPS, MAP_SHORTCUTS
from main import describe_result
from maze_env import Action, CompletedMap, Tile
from maze_view import (
    COLORS, MascotImages, MazeCanvas, draw_ghost_icon, draw_tile_icon, draw_tile_surface, draw_wall_outline,
)


CONTROLLERS = {"수동 조작": "manual", "온라인 DFS": "dfs", "BFS": "bfs", "A* 활용형": "astar"}
MAP_NAMES = {definition.name.replace("x", "×"): map_id for map_id, definition in MAPS.items()}
KEY_ACTIONS = {
    "Up": Action.UP, "Right": Action.RIGHT, "Down": Action.DOWN, "Left": Action.LEFT,
    "w": Action.UP, "d": Action.RIGHT, "s": Action.DOWN, "a": Action.LEFT,
}
DECISION_REASONS = {
    "explore": "BFS로 새 시야를 확보할 가까운 칸 {target} 선택",
    "dfs_advance": "DFS에서 아직 방문하지 않은 이웃 칸 {target} 선택",
    "dfs_backtrack": "갈 수 있는 미방문 이웃이 없어 이전 칸 {target}으로 복귀",
    "dfs_recover": "BFS로 탐색할 길이 남은 갈림길 {target}에 복귀",
    "exhausted": "현재 지도에서 도달할 수 있는 새 탐색 지점 없음",
}


def _decision_reason(decision: Decision, controller: str) -> str:
    if decision.phase in {"key", "exit"}:
        planner = "A*" if controller == "astar" else "BFS"
        goal = "열쇠" if decision.phase == "key" else "출구"
        return f"{planner}로 {goal} {decision.target}까지 경로 계산"
    return DECISION_REASONS[decision.phase].format(target=decision.target)


class MazeApp:
    def __init__(
        self, root: tk.Tk, *, game: ExperimentGame | None = None, map_id: str = "basic15",
        controller: str = "manual", log_dir: Path | str | None = None, max_actions: int = DEFAULT_MAX_ACTIONS,
        maximize: bool = True,
    ) -> None:
        self.root = root
        definition = MAPS[map_id]
        self.game = game or ExperimentGame(
            definition.layout, map_id=definition.map_id, map_name=definition.name,
            controller=controller, log_dir=log_dir, max_actions=max_actions,
        )
        self.running = False
        self._timer: str | None = None
        self._closed = False
        self._save_error = False
        self._completed_map: CompletedMap | None = None
        self._completed_run: str | None = None
        self._view_run_id = self.game.recorder.run_id
        self._progress_zoom = 1.5
        self.show_full_map = tk.BooleanVar(root, False)
        self.show_memory_map = tk.BooleanVar(root, False)
        self.show_path = tk.BooleanVar(root, False)
        self.ghost_choice = tk.BooleanVar(root, False)
        self.trap_choice = tk.BooleanVar(root, self.game.traps_enabled)
        self.board_title = tk.StringVar(root, "현재 시야 3×3")
        self.zoom_text = tk.StringVar(root, "150%")
        self.speed = tk.DoubleVar(root, 1.0)
        self.speed_text = tk.StringVar(root, "1.0×")
        self.map_selection = tk.StringVar(root)
        self.controller_selection = tk.StringVar(root)
        self.message = tk.StringVar(root, "주변을 탐색하세요. 함정은 밟기 전에는 보이지 않습니다.")
        self.saved_message = tk.StringVar(root, "종료 시 JSON·TXT 저장")
        self.fields: dict[str, tk.StringVar] = {}
        self.images = MascotImages(root)
        self._configure_window()
        self._build()
        self.board.zoom = self._progress_zoom
        self.refresh()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<KeyPress>", self._on_key, add="+")
        root.after_idle(self.board.focus_set)
        if maximize:
            try:
                root.state("zoomed")
            except tk.TclError:
                pass  # Other platforms retain the large, resizable window.

    def _configure_window(self) -> None:
        self.root.title("숭실 미로 탐색 — AI 프로젝트")
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width, height = max(900, min(1440, screen_width - 50)), max(620, min(900, screen_height - 75))
        left, top = max(0, (screen_width - width) // 2), max(0, (screen_height - height - 65) // 2)
        self.root.geometry(f"{width}x{height}+{left}+{top}")
        self.root.minsize(min(1000, width), min(700, height))
        self.root.configure(background=COLORS["background"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("맑은 고딕", 10))
        style.configure("TFrame", background=COLORS["background"])
        style.configure("TLabel", background=COLORS["background"], foreground=COLORS["ink"], font=("맑은 고딕", 10))
        style.configure("Muted.TLabel", foreground=COLORS["muted"], font=("맑은 고딕", 9))
        style.configure("Title.TLabel", font=("맑은 고딕", 16, "bold"))
        style.configure("Section.TLabel", foreground="#007da5", font=("맑은 고딕", 11, "bold"))
        style.configure("Value.TLabel", font=("맑은 고딕", 10))
        style.configure("StrongValue.TLabel", font=("맑은 고딕", 10, "bold"))
        style.configure("HeldKey.TLabel", foreground=COLORS["open_exit"], font=("맑은 고딕", 10, "bold"))
        metric_background = "#eef5fb"
        style.configure("Metric.TFrame", background=metric_background)
        style.configure("MetricLabel.TLabel", background=metric_background, foreground="#44566b", font=("맑은 고딕", 9))
        style.configure("MetricValue.TLabel", background=metric_background, font=("맑은 고딕", 9, "bold"))
        style.configure("MetricTotal.TLabel", background=metric_background, foreground="#164e7c", font=("맑은 고딕", 14, "bold"))
        style.configure("TButton", background="#f6f7f8", foreground=COLORS["ink"], padding=(12, 6), borderwidth=1, relief="solid")
        style.map("TButton", background=[("active", "#e9eff3")], foreground=[("disabled", "#a0a6af")])
        style.configure("Accent.TButton", background=COLORS["accent"], foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#008cb7"), ("disabled", "#d4e6ed")], foreground=[("disabled", "#6c8895")])
        style.configure("Zoom.TButton", padding=(6, 3), font=("맑은 고딕", 9))
        style.configure("TCheckbutton", background=COLORS["background"], foreground=COLORS["ink"])
        style.configure("TCombobox", fieldbackground="white", foreground=COLORS["ink"], padding=(6, 4), arrowsize=13)
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")], selectbackground=[("readonly", "#ffffff")],
                  selectforeground=[("readonly", COLORS["ink"])])
        style.configure("Horizontal.TScale", background=COLORS["background"], troughcolor="#e4e7eb")
        icon = self.images.for_size(40)
        if icon is not None:
            self.root.iconphoto(True, icon)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(14, 7, 14, 7))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        self._header_image = self.images.for_size(32)
        if self._header_image is not None:
            ttk.Label(header, image=self._header_image).pack(side="left", padx=(0, 12))
        ttk.Label(header, text="숭실 미로 탐색", style="Title.TLabel").pack(side="left")
        ttk.Separator(header, orient="vertical").pack(side="left", fill="y", padx=20, pady=8)
        ttk.Label(header, text="SSU · AI 프로젝트", style="Muted.TLabel").pack(side="left")
        ttk.Label(header, text="부분 관측 · 내부 지도 기억", style="Muted.TLabel").pack(side="right")
        ttk.Separator(outer).pack(fill="x")

        toolbar = ttk.Frame(outer, padding=(4, 7))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="맵").pack(side="left", padx=(0, 12))
        self.map_combo = ttk.Combobox(toolbar, textvariable=self.map_selection, values=tuple(MAP_NAMES), state="readonly", width=17)
        self.map_combo.pack(side="left")
        self.map_combo.bind("<<ComboboxSelected>>", self._map_selected)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=22)
        ttk.Label(toolbar, text="알고리즘").pack(side="left", padx=(0, 12))
        self.agent_combo = ttk.Combobox(toolbar, textvariable=self.controller_selection, values=tuple(CONTROLLERS), state="readonly", width=17)
        self.agent_combo.pack(side="left")
        self.agent_combo.bind("<<ComboboxSelected>>", self._controller_selected)
        self.trap_toggle = ttk.Checkbutton(
            toolbar, text="고정 함정", variable=self.trap_choice, command=self._toggle_traps,
        )
        self.trap_toggle.pack(side="left", padx=(12, 0))
        self.ghost_toggle = ttk.Checkbutton(
            toolbar, text="순찰 유령", variable=self.ghost_choice, command=self._toggle_ghost,
        )
        self.mode_note = ttk.Label(toolbar, text="", style="Muted.TLabel")
        self.mode_note.pack(side="right")
        ttk.Separator(outer).pack(fill="x")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=(8, 6))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(2, weight=0, minsize=296)
        body.rowconfigure(0, weight=1)
        board_area = ttk.Frame(body)
        board_area.grid(row=0, column=0, sticky="nsew")
        view_toolbar = ttk.Frame(board_area)
        view_toolbar.pack(fill="x", pady=(0, 3))
        view_controls = ttk.Frame(view_toolbar)
        view_controls.pack(side="right")
        ttk.Label(view_toolbar, textvariable=self.board_title, style="Section.TLabel").pack(side="left")
        self.path_toggle = ttk.Checkbutton(view_controls, text="지나온 경로", variable=self.show_path, command=self.refresh)
        self.path_toggle.pack(side="right", padx=(10, 0))
        self.reveal_toggle = ttk.Checkbutton(view_controls, text="전체 지도", variable=self.show_full_map, command=self.refresh)
        self.reveal_toggle.pack(side="right", padx=(10, 0))
        ttk.Button(view_controls, text="내 위치", width=6, style="Zoom.TButton", command=lambda: self.board.center_on_agent()).pack(side="right", padx=(6, 0))
        ttk.Button(view_controls, text="+", width=2, style="Zoom.TButton", command=self.zoom_in).pack(side="right")
        zoom_combo = ttk.Combobox(view_controls, textvariable=self.zoom_text, width=9, state="readonly",
                                 values=("화면 맞춤", "125%", "150%", "175%", "200%", "250%", "300%"))
        zoom_combo.pack(side="right", padx=3)
        zoom_combo.bind("<<ComboboxSelected>>", self._zoom_selected)
        ttk.Button(view_controls, text="−", width=2, style="Zoom.TButton", command=self.zoom_out).pack(side="right")
        viewport = ttk.Frame(board_area)
        viewport.pack(fill="both", expand=True)
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.board = MazeCanvas(viewport, self.images)
        self.board.grid(row=0, column=0, sticky="nsew")
        horizontal = ttk.Scrollbar(viewport, orient="horizontal", command=self.board.xview)
        vertical = ttk.Scrollbar(viewport, orient="vertical", command=self.board.yview)
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        self.board.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.board.bind("<<ZoomIn>>", lambda event: self.zoom_in())
        self.board.bind("<<ZoomOut>>", lambda event: self.zoom_out())
        ttk.Separator(body, orient="vertical").grid(row=0, column=1, sticky="ns", padx=(6, 12))
        sidebar = ttk.Frame(body, width=250)
        sidebar.grid(row=0, column=2, sticky="nsew")
        sidebar.columnconfigure(1, weight=1)
        sidebar.columnconfigure(3, weight=1)
        ttk.Label(sidebar, text="▎ 실행 상태", style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        self._field(sidebar, 1, "status", "상태")
        self.key_value = self._field(sidebar, 1, "key", "열쇠", column=2, style="StrongValue.TLabel")
        self._field(sidebar, 2, "goal", "현재 목표", style="StrongValue.TLabel")
        positions = ttk.Frame(sidebar)
        positions.grid(row=3, column=0, columnspan=4, sticky="w")
        self._field(positions, 0, "position", "위치")
        self._field(positions, 0, "exit", "출구", column=2)
        self._field(sidebar, 4, "decision_action", "방금 선택", style="StrongValue.TLabel")
        self.fields["decision_reason"] = tk.StringVar(self.root)
        ttk.Label(sidebar, textvariable=self.fields["decision_reason"], style="Muted.TLabel", wraplength=280).grid(
            row=5, column=0, columnspan=4, sticky="w", padx=(6, 0),
        )
        ttk.Separator(sidebar).grid(row=6, column=0, columnspan=4, sticky="ew", pady=(5, 5))
        ttk.Label(sidebar, text="▎ 실험 지표", style="Section.TLabel").grid(row=7, column=0, columnspan=4, sticky="w", pady=(0, 5))
        self._build_metrics(sidebar).grid(row=8, column=0, columnspan=4, sticky="ew")
        ttk.Separator(sidebar).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(5, 5))
        ttk.Button(sidebar, text="기록 폴더 열기", command=self.open_logs).grid(row=10, column=0, columnspan=4, sticky="w")
        ttk.Label(sidebar, textvariable=self.saved_message, style="Muted.TLabel", wraplength=280).grid(
            row=11, column=0, columnspan=4, sticky="w", pady=(4, 0),
        )
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x", before=body)
        self.legend = tk.Canvas(footer, height=23, background=COLORS["background"], highlightthickness=0)
        self.legend.pack(fill="x")
        self._draw_legend()
        ttk.Separator(footer).pack(fill="x", pady=(3, 6))

        controls = ttk.Frame(footer)
        controls.pack(fill="x")
        self.run_button = ttk.Button(controls, text="실행", command=self.run, style="Accent.TButton")
        self.run_button.pack(side="left", padx=(0, 10))
        self.pause_button = ttk.Button(controls, text="일시정지", command=self.pause)
        self.pause_button.pack(side="left", padx=(0, 10))
        self.step_button = ttk.Button(controls, text="한 단계", command=self.step)
        self.step_button.pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="초기화", command=self.restart).pack(side="left")
        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(controls, text="속도").pack(side="left", padx=(0, 8))
        ttk.Scale(controls, variable=self.speed, from_=0.25, to=8.0, length=100, command=self._speed_changed).pack(side="left")
        ttk.Label(controls, textvariable=self.speed_text, width=5).pack(side="left", padx=(8, 0))
        self.shortcut_note = ttk.Label(controls, style="Muted.TLabel")
        self.shortcut_note.pack(side="right")
        ttk.Label(footer, textvariable=self.message, style="Muted.TLabel", wraplength=1400).pack(fill="x", pady=(6, 0))

    def _build_metrics(self, parent) -> ttk.Frame:
        metrics = ttk.Frame(parent)
        metrics.columnconfigure((0, 1), weight=1, uniform="totals", minsize=135)
        groups = (
            (("moves", "실제 이동"), ("moves_to_key", "열쇠까지"), ("moves_after_key", "획득 후")),
            (("expanded", "누적 확장"), ("exploration_expanded", "영역 탐색"), ("route_expanded", "목표 경로")),
        )
        for column, group in enumerate(groups):
            card = ttk.Frame(metrics, style="Metric.TFrame", padding=(7, 4))
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (0, 0))
            card.columnconfigure(1, weight=1)
            for row, (key, label) in enumerate(group):
                self.fields[key] = tk.StringVar(self.root)
                ttk.Label(card, text=label, style="MetricLabel.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 5))
                value_style = "MetricTotal.TLabel" if row == 0 else "MetricValue.TLabel"
                ttk.Label(card, textvariable=self.fields[key], style=value_style).grid(row=row, column=1, sticky="e")
        details = ttk.Frame(metrics)
        details.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._field(details, 0, "traps", "함정", pady=1)
        self._field(details, 0, "attempts", "행동 시도", column=2, pady=1)
        self._field(details, 1, "decision_time", "판단 시간", pady=1)
        self._field(details, 1, "route_time", "경로 검색", column=2, pady=1)
        return metrics

    def _field(self, parent, row, key, label, *, column=0, style="Value.TLabel", pady=2) -> ttk.Label:
        self.fields[key] = tk.StringVar(self.root)
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(6, 8), pady=pady)
        value = ttk.Label(parent, textvariable=self.fields[key], style=style, wraplength=150)
        value.grid(row=row, column=column + 1, sticky="w", pady=pady)
        return value

    def _draw_legend(self):
        entries = (
            (Tile.WALL, "벽"), (Tile.FLOOR, "통로"), (None, "미관측"),
            (Tile.START, "시작"), (Tile.KEY, "열쇠"), (Tile.EXIT, "출구"), (Tile.TRAP, "발견한 함정"),
        )
        for index, (tile, label) in enumerate(entries):
            x = index * 103 + 4
            box = (x, 2, x + 19, 21)
            draw_tile_surface(self.legend, tile, box)
            if tile == Tile.WALL:
                draw_wall_outline(self.legend, box)
            elif tile not in (None, Tile.FLOOR):
                draw_tile_icon(self.legend, tile, box)
            self.legend.create_text(x + 25, 12, text=label, anchor="w", fill=COLORS["muted"], font=("맑은 고딕", 9))
        x = len(entries) * 103 + 4
        box = (x, 2, x + 19, 21)
        draw_tile_surface(self.legend, Tile.FLOOR, box, tags="legend-ghost")
        draw_ghost_icon(self.legend, box, tags="legend-ghost")
        self.legend.create_text(x + 25, 12, text="순찰 유령", anchor="w", fill=COLORS["muted"],
                                font=("맑은 고딕", 9), tags="legend-ghost")

    def _speed_changed(self, value):
        self.speed_text.set(f"{float(value):.1f}×")

    def set_zoom(self, zoom: float) -> None:
        self.board.set_zoom(zoom)
        if not self.game.observation.done and self.board.remembered_tiles is None:
            self._progress_zoom = self.board.zoom
        self.zoom_text.set("화면 맞춤" if self.board.zoom == 1.0 else f"{self.board.zoom:.0%}")
        self.board.focus_set()

    def zoom_in(self) -> None:
        self.set_zoom(self.board.zoom + .25)

    def zoom_out(self) -> None:
        self.set_zoom(self.board.zoom - .25)

    def _zoom_selected(self, event=None):
        text = self.zoom_text.get()
        self.set_zoom(1.0 if text == "화면 맞춤" else float(text.rstrip("%")) / 100)

    def _current_goal(self) -> str:
        observation = self.game.observation
        if observation.done:
            return "탈출 완료"
        if observation.has_key:
            return f"출구 {observation.exit_position}"
        if self.game.agent is not None:
            for position, tile in self.game.memory.known_tiles.items():
                if tile == Tile.KEY:
                    return f"열쇠 {position}"
        return "열쇠 탐색"

    def _refresh_board(self) -> None:
        observation = self.game.observation
        run_id = self.game.recorder.run_id
        can_show_memory = self.game.agent is not None and not self.game.finished
        if not can_show_memory or self._view_run_id != run_id:
            self.show_memory_map.set(False)
        self._view_run_id = run_id
        showing_memory = self.show_memory_map.get()
        was_showing_memory = self.board.remembered_tiles is not None
        if observation.done:
            if self._completed_run != run_id:
                self._completed_map = self.game.get_completed_map()
                self._completed_run = run_id
                self.show_full_map.set(True)
                self.show_path.set(True)
                self.board.zoom = 1.0
                self.zoom_text.set("화면 맞춤")
        else:
            if showing_memory and not was_showing_memory:
                self.show_path.set(True)
                self.board.zoom = 1.0
                self.zoom_text.set("화면 맞춤")
            elif self._completed_run is not None or (was_showing_memory and not showing_memory):
                self.board.zoom = self._progress_zoom
                self.zoom_text.set("화면 맞춤" if self.board.zoom == 1.0 else f"{self.board.zoom:.0%}")
            self._completed_map = None
            self._completed_run = None
            self.show_full_map.set(False)
            if not showing_memory:
                self.show_path.set(False)
        map_toggle_variable = self.show_memory_map if can_show_memory else self.show_full_map
        if str(self.reveal_toggle.cget("variable")) != str(map_toggle_variable):
            self.reveal_toggle.configure(
                text="기억한 지도" if can_show_memory else "전체 지도", variable=map_toggle_variable,
            )
        self.reveal_toggle.state(["!disabled"] if can_show_memory or observation.done else ["disabled"])
        showing_full = observation.done and self.show_full_map.get()
        can_show_path = showing_full or showing_memory
        showing_path = can_show_path and self.show_path.get()
        self.path_toggle.state(["!disabled"] if can_show_path else ["disabled"])
        if observation.done and not self._save_error:
            if showing_path:
                self.message.set("파란 선은 실제 이동입니다. 반복 이동은 진하게, 시작점 복귀는 끊어서 표시합니다.")
            elif showing_full:
                self.message.set("탈출 성공! 열쇠는 처음 놓였던 위치로 표시합니다.")
            else:
                self.message.set("탈출 성공!")
        if showing_full:
            title = "전체 지도"
        elif showing_memory:
            title = "기억한 지도"
        else:
            title = "현재 시야 3×3"
        self.board_title.set(title)
        self.board.render(
            observation, self._completed_map if showing_full else None,
            remembered_tiles=self.game.memory.observed_tiles() if showing_memory else None,
            travel_edges=self.game.recorder.travel_edges if showing_path else (),
            recenter=was_showing_memory and not showing_memory and not observation.done,
        )

    def refresh(self) -> None:
        observation = self.game.observation
        decision = self.game.last_decision
        recorder = self.game.recorder
        metrics = recorder.computation
        self.map_selection.set(self.game.map_name.replace("x", "×"))
        self.controller_selection.set(next(name for name, code in CONTROLLERS.items() if code == self.game.controller))
        if self._save_error:
            status = "저장 실패"
        elif observation.done:
            status = "탈출 성공"
        elif self.game.end_reason is not None:
            status = {EndReason.ACTION_LIMIT: "행동 한도 종료", EndReason.EXHAUSTED: "탐색 종료"}.get(self.game.end_reason, "실험 종료")
        elif self.game.agent is None:
            status = "수동 조작"
        else:
            status = "실행 중" if self.running else "일시정지"
        chosen_action = "—" if self.game.agent is None else "탐색 대기"
        reason = ""
        if decision is not None:
            if decision.action is None:
                chosen_action = "이동 없음"
            else:
                chosen_action = f"{ACTION_LABELS[decision.action.value]} 이동"
            reason = f"{_decision_reason(decision, self.game.controller)} · 확장 {decision.stats.expanded_nodes:,}개"
        moves_to_key = recorder.moves_to_first_key
        moves_after_key = recorder.moves_after_first_key
        key_progress = "미획득" if self.game.finished else "찾는 중"
        values = {
            "status": status,
            "decision_action": chosen_action, "decision_reason": reason,
            "position": str(observation.position), "goal": self._current_goal(),
            "exit": str(observation.exit_position), "key": "보유" if observation.has_key else "미보유",
            "moves": f"{observation.moves:,}회",
            "moves_to_key": f"{moves_to_key:,}회" if moves_to_key is not None else key_progress,
            "moves_after_key": f"{moves_after_key:,}회" if moves_after_key is not None else "—",
            "expanded": f"{metrics['expanded_nodes']:,}개",
            "exploration_expanded": f"{metrics['exploration_expanded_nodes']:,}개",
            "route_expanded": f"{metrics['route_expanded_nodes']:,}개",
            "traps": f"{observation.trap_hits:,}회",
            "decision_time": f"{metrics['decision_time_seconds'] * 1_000:,.2f} ms",
            "route_time": f"{metrics['path_planning_seconds'] * 1_000:,.2f} ms",
            "attempts": f"{recorder.action_attempts:,}" + (f" / {self.game.max_actions:,}" if self.game.agent is not None else ""),
        }
        for key, value in values.items():
            self.fields[key].set(value)
        self.key_value.configure(style="HeldKey.TLabel" if observation.has_key else "StrongValue.TLabel")
        self._refresh_ghost_controls()
        self.run_button.state(["disabled"] if self.game.finished or self.running else ["!disabled"])
        self.pause_button.state(["!disabled"] if self.running else ["disabled"])
        can_step = (self.game.agent is not None or self.game.ghost_enabled) and not self.game.finished
        self.step_button.state(["!disabled"] if can_step else ["disabled"])
        self._refresh_board()

    def _can_use_ghost(self) -> bool:
        return self.game.map_id == "hard21" and self.game.controller == "manual"

    def _refresh_ghost_controls(self) -> None:
        self.trap_choice.set(self.game.traps_enabled)
        enabled = self.game.ghost_enabled
        self.ghost_choice.set(enabled)
        if self._can_use_ghost():
            if not self.ghost_toggle.winfo_manager():
                self.ghost_toggle.pack(side="left", padx=(16, 0), before=self.mode_note)
        else:
            self.ghost_toggle.pack_forget()
        self.legend.itemconfigure("legend-ghost", state="normal" if enabled else "hidden")
        self.step_button.configure(text="대기" if enabled else "한 단계")
        shortcuts = "방향키/WASD · Space 대기" if enabled else "방향키/WASD · N 단계 · P 실행/정지"
        self.shortcut_note.configure(text=shortcuts)
        if enabled:
            observation = self.game.observation
            note = f"턴 {observation.turns:,} · 유령 접촉 {observation.ghost_hits:,}"
        elif self.game.controller == "astar":
            note = "미지 영역 BFS · 목표 경로 A*"
        else:
            note = "좌표 (행, 열) · 0부터 시작"
        self.mode_note.configure(text=note)

    def _toggle_traps(self) -> None:
        enabled = self.trap_choice.get()
        self._pause_timer()
        if enabled != self.game.traps_enabled:
            if self._operation(lambda: self.game.change_traps(enabled)):
                self.saved_message.set("이전 기록 저장 완료")
                self.message.set("고정 함정을 켜고 새 실험을 시작했습니다." if enabled
                                 else "고정 함정을 끄고 새 실험을 시작했습니다.")
        self.refresh()
        self.board.focus_set()

    def _toggle_ghost(self) -> None:
        enabled = self.ghost_choice.get()
        if self._can_use_ghost() and enabled != self.game.ghost_enabled:
            patrol = HARD21_GHOST_PATROL if enabled else ()
            if self._operation(lambda: self.game.change_ghost_patrol(patrol)):
                self.saved_message.set("이전 기록 저장 완료")
                self.message.set("순찰 유령을 켰습니다. Space로 한 턴 기다릴 수 있습니다."
                                 if enabled else "순찰 유령을 끄고 새 실험을 시작했습니다.")
        self.refresh()
        self.board.focus_set()

    def _pause_timer(self) -> None:
        self.running = False
        if self._timer is not None:
            self.root.after_cancel(self._timer)
            self._timer = None

    def _operation(self, operation) -> bool:
        try:
            operation()
            self._save_error = False
            return True
        except OSError as error:
            self._pause_timer()
            self._save_error = True
            self.message.set("기록을 저장하지 못했습니다. 현재 실험을 유지합니다. 초기화 또는 창 닫기로 다시 저장할 수 있습니다.")
            self.refresh()
            messagebox.showerror("기록 저장 실패", f"{error}\n\n현재 실험은 유지됩니다. 저장 폴더를 확인한 뒤 다시 시도하세요.", parent=self.root)
            return False

    def _describe_saved(self) -> None:
        if self.game.last_saved:
            self.saved_message.set("저장 완료 · JSON / TXT")

    def _one_action(self) -> None:
        result = self.game.automatic_step()
        self.message.set(describe_result(result) if result is not None else REASON_LABELS[self.game.end_reason])
        self._describe_saved()
        if self.game.finished:
            self._pause_timer()

    def step(self) -> None:
        if self.game.ghost_enabled:
            self.move(Action.WAIT)
            self.board.focus_set()
            return
        self._pause_timer()
        if self.game.agent is not None and not self.game.finished:
            self._operation(self._one_action)
        self.refresh()
        self.board.focus_set()

    def run(self) -> None:
        if self.game.finished:
            return
        if self.game.agent is None:
            self.message.set("수동 조작: 방향키 또는 WASD로 한 칸씩 이동하세요.")
        elif not self.running:
            self.running = True
            self.message.set("자동 실행 중입니다. 일시정지하거나 N으로 한 단계씩 확인할 수 있습니다.")
            self._timer = self.root.after(0, self._tick)
        self.refresh()
        self.board.focus_set()

    def _tick(self) -> None:
        self._timer = None
        if self._closed or not self.running:
            return
        self._operation(self._one_action)
        self.refresh()
        if self.running and not self.game.finished:
            interval = max(10, round(250 / max(.25, self.speed.get())))
            self._timer = self.root.after(interval, self._tick)

    def pause(self) -> None:
        self._pause_timer()
        self.message.set("일시정지했습니다. 한 단계 버튼 또는 N으로 다음 행동을 확인할 수 있습니다.")
        self.refresh()
        self.board.focus_set()

    def move(self, action: Action) -> None:
        if self.game.agent is not None or self.game.finished:
            return
        def operation():
            self.message.set(describe_result(self.game.move(action)))
            self._describe_saved()
        self._operation(operation)
        self.refresh()

    def restart(self) -> None:
        self._pause_timer()
        if self._operation(self.game.restart):
            self.saved_message.set("이전 기록 저장 완료")
            self.message.set("새 실험입니다. 내부 지도, 열쇠, 누적 기록을 초기화했습니다.")
        self.refresh()
        self.board.focus_set()

    def select_map(self, map_id: str) -> None:
        self._pause_timer()
        if map_id != self.game.map_id:
            selected = MAPS[map_id]
            if self._operation(lambda: self.game.change_map(selected.layout, selected.map_id, selected.name)):
                self.saved_message.set("이전 기록 저장 완료")
                self.message.set(selected.description)
        self.refresh()
        self.board.focus_set()

    def select_controller(self, controller: str) -> None:
        self._pause_timer()
        if controller != self.game.controller:
            if self._operation(lambda: self.game.change_controller(controller)):
                self.saved_message.set("이전 기록 저장 완료")
                self.message.set("실행 주체를 바꾸고 새 실험을 시작했습니다.")
        self.refresh()
        self.board.focus_set()

    def _map_selected(self, event=None):
        self.select_map(MAP_NAMES[self.map_selection.get()])

    def _controller_selected(self, event=None):
        self.select_controller(CONTROLLERS[self.controller_selection.get()])

    def _on_key(self, event):
        if event.widget.winfo_class() in {"TCombobox", "TEntry", "Entry", "TScale", "TButton", "TCheckbutton", "TScrollbar", "Button", "Spinbox", "TSpinbox"}:
            return None
        key = event.keysym if event.keysym in KEY_ACTIONS else event.keysym.lower()
        if getattr(event, "state", 0) & 0x4:
            if key in {"plus", "equal", "kp_add"}:
                self.zoom_in()
                return "break"
            if key in {"minus", "kp_subtract"}:
                self.zoom_out()
                return "break"
            if key in {"0", "kp_0"}:
                self.set_zoom(1.0)
                return "break"
        if key in KEY_ACTIONS:
            self.move(KEY_ACTIONS[key])
        elif key in {"n", "space", "return"}:
            self.step()
        elif key == "p" and self.game.agent is not None:
            self.pause() if self.running else self.run()
        elif key == "r":
            self.restart()
        elif key in MAP_SHORTCUTS:
            self.select_map(MAP_SHORTCUTS[key])
        elif key in {"4", "5", "6", "m"}:
            self.select_controller({"4": "dfs", "5": "bfs", "6": "astar", "m": "manual"}[key])
        elif key == "q":
            self.close()
        else:
            return None
        return "break"

    def open_logs(self) -> None:
        directory = self.game.log_dir.resolve()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(directory))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(directory)])
        except OSError as error:
            messagebox.showerror("폴더 열기 실패", str(error), parent=self.root)

    def close(self, reason: EndReason = EndReason.QUIT) -> bool:
        self._pause_timer()
        if not self._operation(lambda: self.game.finish(reason)):
            return False
        self._closed = True
        self.root.destroy()
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="숭실 미로 탐색 — Python/Tkinter GUI")
    parser.add_argument("--map", choices=tuple(MAPS), default="basic15", dest="map_id")
    parser.add_argument("--agent", choices=tuple(CONTROLLERS.values()), default="manual")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    args = parser.parse_args(argv)
    if args.max_actions < 1:
        parser.error("--max-actions는 1 이상이어야 합니다.")
    root = tk.Tk()
    app = MazeApp(root, map_id=args.map_id, controller=args.agent, log_dir=args.log_dir, max_actions=args.max_actions)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close(EndReason.INTERRUPTED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
