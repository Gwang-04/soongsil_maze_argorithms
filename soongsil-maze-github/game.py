"""Shared experiment lifecycle for manual input and public-information agents."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from agents import ALGORITHM_LABELS, Decision, SearchAgent, agent_settings
from experiment_log import EndReason, RunRecorder, SavedRun
from maze_env import Action, AgentMemory, CompletedMap, Event, EventKind, MazeEnv, MazeLayout, Position, StepResult


DEFAULT_MAX_ACTIONS = 5000


class ExperimentGame:
    def __init__(
        self, layout: MazeLayout | None = None, *, log_dir: Path | str | None = None,
        map_id: str = "basic15", map_name: str = "기본 15x15", controller: str = "manual",
        max_actions: int = DEFAULT_MAX_ACTIONS, capture_tasks: bool = False, traps_enabled: bool = True,
    ) -> None:
        if controller not in ALGORITHM_LABELS:
            raise ValueError(f"Unknown controller: {controller}")
        if max_actions < 1:
            raise ValueError("max_actions must be positive.")
        self._layout = layout or MazeLayout()
        self.traps_enabled = traps_enabled
        self._ghost_patrol: tuple[Position, ...] = ()
        self.map_id, self.map_name = map_id, map_name
        self.controller, self.max_actions = controller, max_actions
        self.capture_tasks = capture_tasks
        self.log_dir = Path(log_dir) if log_dir is not None else Path(__file__).resolve().parent / "logs"
        self._new_run()

    def _new_run(self, env: MazeEnv | None = None) -> None:
        self.last_saved: SavedRun | None = None
        self.end_reason: EndReason | None = None
        self.last_decision: Decision | None = None
        layout = self._active_layout()
        self.env = env if env is not None else MazeEnv(layout, ghost_patrol=self._ghost_patrol)
        self.observation = self.env.get_observation()
        self.memory = AgentMemory()
        self.memory.update(self.observation)
        self.agent = None if self.controller == "manual" else SearchAgent(self.controller, capture_tasks=self.capture_tasks)
        settings = {"algorithm": "manual", "max_action_attempts": None} if self.agent is None else agent_settings(self.controller, self.max_actions)
        settings["trap_return_keeps_key"] = True
        settings["traps_enabled"] = self.traps_enabled
        if self.ghost_enabled:
            settings["ghost_enabled"] = True
        self.recorder = RunRecorder(
            layout, self.map_id, self.map_name, self.observation, self.memory,
            controller=self.controller, settings=settings,
        )

    def _active_layout(self) -> MazeLayout:
        if self.traps_enabled:
            return self._layout
        return MazeLayout(tuple(row.replace("T", ".") for row in self._layout.rows))

    def change_traps(self, enabled: bool) -> SavedRun:
        saved = self.finish(EndReason.TRAP_MODE_CHANGED)
        self.traps_enabled = enabled
        self._new_run()
        return saved

    @property
    def finished(self) -> bool:
        return self.end_reason is not None or self.observation.done

    @property
    def ghost_enabled(self) -> bool:
        return bool(self._ghost_patrol)

    def get_completed_map(self) -> CompletedMap:
        return self.env.get_completed_map()

    def move(self, action: Action | str, *, decision: dict | None = None) -> StepResult:
        if self.finished:
            return StepResult(self.observation, (Event(EventKind.ALREADY_FINISHED, self.observation.position),))
        result = self.env.step(action)
        self.observation = result.observation
        self.memory.update(self.observation, result.events)
        if self.agent is not None:
            self.agent.observe(result)
        self.recorder.record(action, result, decision=decision)
        if result.done:
            self.finish(EndReason.ESCAPED)
        return result

    def automatic_step(self) -> StepResult | None:
        if self.finished:
            return None
        if self.agent is None:
            raise ValueError("Select an automatic agent first.")
        if self.recorder.action_attempts >= self.max_actions:
            self.finish(EndReason.ACTION_LIMIT)
            return None
        began = perf_counter()
        decision = self.agent.choose(self.observation, self.memory)
        decision_seconds = perf_counter() - began
        self.last_decision = decision
        details = {
            **asdict(decision.stats), "decision_time_seconds": decision_seconds,
            "phase": decision.phase, "target": list(decision.target) if decision.target is not None else None,
        }
        self.recorder.record_decision(details)
        if decision.action is None:
            self.finish(EndReason.EXHAUSTED)
            return None
        result = self.move(decision.action, decision=details)
        if not self.finished and self.recorder.action_attempts >= self.max_actions:
            self.finish(EndReason.ACTION_LIMIT)
        return result

    def finish(self, reason: EndReason | str) -> SavedRun:
        saved = self.recorder.finish(reason, self.observation, self.memory, self.log_dir)
        self.last_saved, self.end_reason = saved, saved.reason
        return saved

    def restart(self) -> SavedRun:
        saved = self.finish(EndReason.RESET)
        self._new_run()
        return saved

    def change_map(self, layout: MazeLayout, map_id: str, map_name: str) -> SavedRun:
        saved = self.finish(EndReason.MAP_CHANGED)
        self._layout, self.map_id, self.map_name = layout, map_id, map_name
        self._ghost_patrol = ()
        self._new_run()
        return saved

    def change_controller(self, controller: str) -> SavedRun:
        if controller not in ALGORITHM_LABELS:
            raise ValueError(f"Unknown controller: {controller}")
        saved = self.finish(EndReason.CONTROLLER_CHANGED)
        self.controller = controller
        self._ghost_patrol = ()
        self._new_run()
        return saved

    def change_ghost_patrol(self, patrol: tuple[Position, ...]) -> SavedRun:
        if self.controller != "manual":
            raise ValueError("Ghost patrol is available in manual mode.")
        patrol = tuple(patrol)
        env = MazeEnv(self._active_layout(), ghost_patrol=patrol)
        saved = self.finish(EndReason.GHOST_MODE_CHANGED)
        self._ghost_patrol = patrol
        self._new_run(env)
        return saved
