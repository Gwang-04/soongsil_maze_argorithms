"""GUI boundaries: local visibility, exact sprite placement and run lifecycle."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import patch

from game import ExperimentGame
from gui import MazeApp
from maze_env import Action, CompletedMap, MazeLayout
from maze_view import ASSET_PATH, COLORS, GridGeometry, MascotImages


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = (ROOT / "tmp" / "test-runs").resolve()
FIXTURE = MazeLayout((
    "#########", "#S..E...#", "#.......#", "#..K.T..#",
    "#.......#", "#.......#", "#########",
))


class GeometryTests(unittest.TestCase):
    def test_both_map_sizes_use_square_cells_and_every_sprite_stays_in_its_cell(self):
        for shape in ((15, 15), (21, 21)):
            for width, height in ((780, 520), (610, 410), (1000, 690)):
                with self.subTest(shape=shape, viewport=(width, height)):
                    geometry = GridGeometry.fit(width, height, shape)
                    self.assertGreater(geometry.cell_size, 10)
                    self.assertGreaterEqual(geometry.x, 30)
                    self.assertGreaterEqual(geometry.y, 25)
                    self.assertLessEqual(geometry.x + shape[1] * geometry.cell_size, width)
                    self.assertLessEqual(geometry.y + shape[0] * geometry.cell_size, height)
                    for position in ((0, 0), (1, 1), (7, 7), (shape[0] - 1, shape[1] - 1)):
                        x, y, right, bottom = geometry.bounds(position)
                        sx, sy, sr, sb = geometry.sprite_bounds(position)
                        self.assertEqual(right - x, bottom - y)
                        self.assertGreater(sx, x)
                        self.assertGreater(sy, y)
                        self.assertLess(sr, right)
                        self.assertLess(sb, bottom)
                        self.assertAlmostEqual(sr - sx, geometry.cell_size * .7)


class GuiTests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="maze-gui-test-", dir=TEMP_ROOT)
        self.directory = Path(self.temporary.name)
        self.root = tk.Tk()
        # Actual Tk layout is exercised without flashing visible test windows.
        self.root.attributes("-alpha", 0.0)
        game = ExperimentGame(FIXTURE, map_id="fixture", map_name="테스트 맵", log_dir=self.directory)
        self.app = MazeApp(self.root, game=game, maximize=False)
        self.root.update()

    def tearDown(self):
        if not self.app._closed:
            self.app.game.log_dir = self.directory
            self.app.close()
        self.assertTrue(self.directory.resolve().is_relative_to(TEMP_ROOT))
        self.temporary.cleanup()

    def test_canvas_uses_only_current_observation_even_after_cells_are_remembered(self):
        for action in (Action.DOWN, Action.DOWN, Action.RIGHT):
            self.app.move(action)
        observation = self.app.game.observation
        self.assertIn((1, 1), self.app.game.memory.known_tiles)
        self.assertNotIn((1, 1), {cell.position for cell in observation.visible})
        ids = self.app.board.find_withtag("cell:1:1")
        self.assertEqual(len(ids), 1)
        self.assertIn("masked", self.app.board.gettags(ids[0]))
        self.assertEqual(len(self.app.board.find_withtag("visible")), len(observation.visible))
        self.assertEqual(len(self.app.board.find_withtag("unknown")), 9 * 7 - len(observation.visible))

    def test_untriggered_trap_is_never_drawn_and_revealed_trap_obeys_local_view(self):
        for action in (Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.RIGHT):
            self.app.move(action)
        self.assertEqual(self.app.game.observation.position, (3, 4))
        self.assertEqual(self.app.board.find_withtag("icon:T"), ())
        self.app.move(Action.RIGHT)
        self.assertIn((3, 5), self.app.game.memory.known_traps)
        self.assertEqual(self.app.board.find_withtag("icon:T"), ())
        for action in (Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.RIGHT):
            self.app.move(action)
        self.assertTrue(self.app.board.find_withtag("icon:T"))

    def test_rendered_character_and_highlight_identify_exactly_one_cell(self):
        for map_id in ("basic15", "hard21"):
            with self.subTest(map_id=map_id):
                self.app.select_map(map_id)
                self.root.update()
                self.app.move(Action.RIGHT)
                geometry = self.app.board.geometry_info
                position = self.app.game.observation.position
                x, y, right, bottom = geometry.bounds(position)
                sx, sy, sr, sb = self.app.board.bbox("agent")
                self.assertGreater(sx, x)
                self.assertGreater(sy, y)
                self.assertLess(sr, right)
                self.assertLess(sb, bottom)
                self.assertEqual(len(self.app.board.find_withtag("agent-cell")), 1)
                self.assertEqual(self.app.fields["position"].get(), str(position))
                self.assertLessEqual(self.app.board._agent_image.width(), int(geometry.cell_size * .7))
                self.assertLessEqual(self.app.board._agent_image.height(), int(geometry.cell_size * .7))

    def test_mascot_has_transparency_and_standard_tk_fallback_needs_no_pillow(self):
        source = tk.PhotoImage(master=self.root, file=str(ASSET_PATH))
        self.assertTrue(source.transparency_get(0, 0))
        images = MascotImages(self.root, smooth=False)
        for size in (12, 21, 40):
            image = images.for_size(size)
            self.assertIsNotNone(image)
            self.assertLessEqual(image.width(), size)
            self.assertLessEqual(image.height(), size)

    def test_fixed_trap_toggle_resets_persists_and_restores_original_layout(self):
        self.app.move(Action.RIGHT)
        self.app.trap_toggle.invoke()
        self.assertFalse(self.app.game.traps_enabled)
        self.assertEqual(self.app.game.observation.moves, 0)
        self.assertFalse(any("T" in row for row in self.app.game.env._layout.rows))
        previous = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(previous["termination"]["reason"], "trap_mode_changed")
        self.assertTrue(previous["settings"]["traps_enabled"])
        self.app.select_controller("bfs")
        self.app.restart()
        self.assertFalse(self.app.game.traps_enabled)
        self.app.trap_toggle.invoke()
        self.assertTrue(self.app.game.traps_enabled)
        self.assertTrue(any("T" in row for row in self.app.game.env._layout.rows))

    def test_trap_toggle_save_failure_preserves_active_run(self):
        run_id = self.app.game.recorder.run_id
        with patch.object(self.app.game, "finish", side_effect=OSError("test")), patch("gui.messagebox.showerror"):
            self.app.trap_toggle.invoke()
        self.assertTrue(self.app.game.traps_enabled)
        self.assertTrue(self.app.trap_choice.get())
        self.assertEqual(self.app.game.recorder.run_id, run_id)

    def test_single_step_runs_one_action_and_pause_cancels_scheduled_actions(self):
        self.app.select_controller("bfs")
        self.app.step_button.invoke()
        self.assertEqual(self.app.game.recorder.action_attempts, 1)
        self.assertFalse(self.app.running)
        self.app.run_button.invoke()
        self.root.update()
        self.assertTrue(self.app.running)
        self.app.pause_button.invoke()
        count = self.app.game.recorder.action_attempts
        self.root.after(70, self.root.quit)
        self.root.mainloop()
        self.assertFalse(self.app.running)
        self.assertIsNone(self.app._timer)
        self.assertEqual(self.app.game.recorder.action_attempts, count)
        self.app.step_button.invoke()
        self.assertEqual(self.app.game.recorder.action_attempts, count + 1)

    def test_hard_map_patrol_option_waits_in_manual_mode_and_clears_for_agents(self):
        self.assertFalse(self.app.ghost_toggle.winfo_manager())
        self.app.select_map("hard21")
        self.assertEqual(self.app.ghost_toggle.winfo_manager(), "pack")
        self.app.ghost_toggle.invoke()
        self.assertTrue(self.app.game.ghost_enabled)
        self.assertEqual(self.app.step_button.cget("text"), "대기")
        self.assertFalse(self.app.board.find_withtag("ghost"))
        self.app.step_button.invoke()
        self.app._on_key(SimpleNamespace(widget=self.app.board, keysym="space"))
        self.assertEqual(self.app.game.observation.turns, 2)
        self.assertEqual(self.app.game.observation.moves, 0)
        self.assertIn("턴 2", self.app.mode_note.cget("text"))

        self.app.restart()
        self.assertTrue(self.app.game.ghost_enabled)
        self.assertEqual(self.app.game.observation.turns, 0)
        self.app.select_controller("bfs")
        self.assertFalse(self.app.game.ghost_enabled)
        self.assertFalse(self.app.ghost_toggle.winfo_manager())
        self.assertEqual(self.app.step_button.cget("text"), "한 단계")
        self.app.step_button.invoke()
        self.assertEqual(self.app.game.observation.moves, 1)

    def test_all_three_agents_finish_and_ui_metrics_match_saved_records(self):
        for algorithm in ("dfs", "bfs", "astar"):
            with self.subTest(algorithm=algorithm):
                self.app.select_controller(algorithm)
                self.app.reveal_toggle.invoke()
                for _ in range(150):
                    if self.app.game.finished:
                        break
                    self.app.step_button.invoke()
                self.assertTrue(self.app.game.observation.done)
                self.assertIsNone(self.app.board.remembered_tiles)
                self.assertTrue(self.app.show_full_map.get())
                self.assertEqual(self.app.reveal_toggle.cget("text"), "전체 지도")
                self.assertEqual(self.app.fields["status"].get(), "탈출 성공")
                saved = self.app.game.last_saved
                data = json.loads(saved.json_path.read_text(encoding="utf-8"))
                self.assertEqual(data["controller"], algorithm)
                self.assertEqual(self.app.fields["moves"].get(), f"{data['summary']['moves']:,}회")
                self.assertEqual(self.app.fields["moves_to_key"].get(), f"{data['summary']['moves_to_key']:,}회")
                self.assertEqual(self.app.fields["moves_after_key"].get(), f"{data['summary']['moves_after_key']:,}회")
                self.assertEqual(self.app.fields["expanded"].get(), f"{data['computation']['expanded_nodes']:,}개")
                self.assertEqual(self.app.fields["exploration_expanded"].get(), f"{data['computation']['exploration_expanded_nodes']:,}개")
                self.assertEqual(self.app.fields["route_expanded"].get(), f"{data['computation']['route_expanded_nodes']:,}개")
                self.assertTrue(saved.text_path.is_file())
                self.assertIn("저장 완료", self.app.saved_message.get())
                self.assertIn("disabled", self.app.run_button.state())

    def test_astar_step_shows_the_selected_method_and_updates_goal_on_key_pickup(self):
        self.app.select_controller("astar")
        fields = self.app.fields
        self.assertEqual(fields["goal"].get(), "열쇠 탐색")
        self.assertEqual(fields["key"].get(), "미보유")
        self.assertEqual(fields["moves_to_key"].get(), "찾는 중")
        self.assertEqual(fields["moves_after_key"].get(), "—")
        self.app.step_button.invoke()
        self.assertEqual(fields["decision_action"].get(), "오른쪽 이동")
        self.assertIn("BFS로", fields["decision_reason"].get())
        self.assertIn("(1, 2)", fields["decision_reason"].get())

        for _ in range(30):
            if self.app.game.observation.has_key:
                break
            self.app.step_button.invoke()
        self.assertTrue(self.app.game.observation.has_key)
        self.assertEqual(fields["moves_to_key"].get(), f"{self.app.game.observation.moves}회")
        self.assertEqual(fields["moves_after_key"].get(), "0회")
        self.assertEqual(fields["key"].get(), "보유")
        self.assertEqual(fields["goal"].get(), "출구 (1, 4)")
        self.assertIn("A*로 열쇠 (3, 3)까지", fields["decision_reason"].get())

        self.app.restart()
        self.assertEqual(fields["goal"].get(), "열쇠 탐색")
        self.assertEqual(fields["moves_to_key"].get(), "찾는 중")
        self.assertEqual(fields["decision_action"].get(), "탐색 대기")
        self.assertEqual(fields["decision_reason"].get(), "")

    def test_controller_map_and_reset_save_before_clearing_the_current_experiment(self):
        self.app.move(Action.RIGHT)
        first_id = self.app.game.recorder.run_id
        self.app.select_controller("dfs")
        self.app.step()
        self.app.select_map("hard21")
        self.app.step()
        self.app.restart()
        self.assertEqual(self.app.game.observation.moves, 0)
        self.assertEqual(self.app.game.memory.travel_history, [(1, 1)])
        self.assertEqual(self.app.fields["expanded"].get(), "0개")
        logs = [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")]
        self.assertEqual(len(logs), 3)
        self.assertEqual({record["termination"]["reason"] for record in logs}, {"controller_changed", "map_changed", "reset"})
        first = next(record for record in logs if record["run_id"] == first_id)
        self.assertEqual(first["summary"]["moves"], 1)

    def test_keyboard_does_not_steal_arrow_keys_from_dropdowns(self):
        event = SimpleNamespace(widget=self.app.map_combo, keysym="Right")
        self.assertIsNone(self.app._on_key(event))
        self.assertEqual(self.app.game.observation.moves, 0)
        self.app._on_key(SimpleNamespace(widget=self.app.board, keysym="Right"))
        self.assertEqual(self.app.game.observation.moves, 1)

    def test_save_error_keeps_window_and_current_state_for_retry(self):
        self.app.move(Action.RIGHT)
        before = self.app.game.observation
        blocker = self.directory / "not-a-folder"
        blocker.write_text("block", encoding="utf-8")
        self.app.game.log_dir = blocker
        with patch("gui.messagebox.showerror") as dialog:
            self.assertFalse(self.app.close())
            dialog.assert_called_once()
        self.assertTrue(self.root.winfo_exists())
        self.assertEqual(self.app.game.observation, before)
        self.assertEqual(self.app.fields["status"].get(), "저장 실패")
        self.app.game.log_dir = self.directory
        self.assertTrue(self.app.close())

    def test_controls_and_sidebar_fit_a_small_window_with_a_decision(self):
        self.root.minsize(900, 620)
        self.root.geometry("900x620")
        self.app.select_controller("astar")
        self.app.step_button.invoke()
        self.root.update()
        bottom = self.root.winfo_rooty() + self.root.winfo_height()
        for widget in (self.app.run_button, self.app.pause_button, self.app.step_button):
            self.assertTrue(widget.winfo_ismapped())
            self.assertLess(widget.winfo_rooty() + widget.winfo_height(), bottom)
        sidebar = self.app.key_value.master
        sidebar_bottom = sidebar.winfo_rooty() + sidebar.winfo_height()
        for widget in sidebar.winfo_children():
            self.assertTrue(widget.winfo_ismapped())
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), sidebar_bottom)
        self.assertGreater(self.app.board.winfo_width(), 400)
        self.assertGreater(self.app.board.winfo_height(), 300)

    def finish_manual_fixture(self):
        for action in (Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.UP, Action.UP, Action.RIGHT):
            self.app.move(action)
        self.assertTrue(self.app.game.observation.done)

    def assert_local_view(self):
        self.assertFalse(self.app.show_full_map.get())
        self.assertFalse(self.app.show_memory_map.get())
        self.assertIsNone(self.app.board.completed_map)
        self.assertIsNone(self.app.board.remembered_tiles)
        self.assertTrue(self.app.path_toggle.instate(["disabled"]))
        self.assertEqual(len(self.app.board.find_withtag("visible")), len(self.app.game.observation.visible))
        self.assertGreater(len(self.app.board.find_withtag("unknown")), 0)

    def test_remembered_map_shows_observed_tiles_and_actual_path_during_automatic_steps(self):
        self.assertTrue(self.app.reveal_toggle.instate(["disabled"]))
        self.app.select_controller("bfs")
        self.app.set_zoom(2.0)
        self.assert_local_view()
        self.assertEqual(self.app.reveal_toggle.cget("text"), "기억한 지도")

        self.app.reveal_toggle.invoke()
        board = self.app.board
        self.assertTrue(self.app.show_memory_map.get())
        self.assertEqual(board.zoom, 1.0)
        self.assertIn((1, 4), self.app.game.memory.known_tiles)
        self.assertNotIn((1, 4), board.remembered_tiles)
        self.assertFalse(board.find_withtag("icon:E"))

        self.app.step_button.invoke()
        self.app.step_button.invoke()
        local = {cell.position for cell in self.app.game.observation.visible}
        self.assertNotIn((1, 1), local)
        self.assertTrue(board.find_withtag("icon:S"))
        self.assertTrue(board.find_withtag("icon:E"))
        self.assertFalse(board.find_withtag("icon:K"))
        self.assertFalse(board.find_withtag("icon:T"))
        self.assertGreater(len(board.find_withtag("visible")), len(local))
        self.assertEqual(len(board.find_withtag("travel-path")), 2)

        self.app.path_toggle.invoke()
        self.app.step_button.invoke()
        self.assertFalse(board.find_withtag("travel-path"))
        self.app.path_toggle.invoke()
        self.app.set_zoom(2.5)
        self.root.geometry("1100x740")
        self.root.update()
        self.assertTrue(board.find_withtag("icon:S"))
        self.assertTrue(board.find_withtag("travel-path"))

        self.app.reveal_toggle.invoke()
        self.assert_local_view()
        self.assertEqual(board.zoom, 2.0)
        self.assertFalse(board.find_withtag("travel-path"))
        self.app.reveal_toggle.invoke()
        self.app.restart()
        self.assert_local_view()
        self.assertEqual(board.zoom, 2.0)

    def test_zoom_enlarges_cells_without_revealing_extra_information(self):
        self.app.select_map("hard21")
        self.root.update()
        self.app.set_zoom(1.0)
        cell = self.app.board.geometry_info.cell_size
        before = self.app.game.observation
        self.app.set_zoom(2.0)
        self.assertEqual(self.app.board.geometry_info.cell_size, cell * 2)
        self.assertEqual(self.app.game.observation, before)
        self.assertEqual(len(self.app.board.find_withtag("visible")), 9)
        self.assertEqual(len(self.app.board.find_withtag("unknown")), 21 * 21 - 9)
        self.assertEqual(self.app.zoom_text.get(), "200%")
        self.app.set_zoom(1.0)
        self.assertEqual(self.app.board.geometry_info.cell_size, cell)
        self.assertEqual(self.app.zoom_text.get(), "화면 맞춤")

    def test_zoomed_camera_keeps_the_agent_in_view_after_movement(self):
        self.app.select_map("basic15")
        self.root.update()
        self.app.set_zoom(3.0)
        for _ in range(12):
            self.app.move(Action.RIGHT)
        for _ in range(11):
            self.app.move(Action.DOWN)
        self.assertEqual(self.app.game.observation.position, (12, 13))
        self.assertGreater(self.app.board.yview()[0], 0)
        cx, cy = self.app.board.geometry_info.center(self.app.game.observation.position)
        screen_x = cx - self.app.board.canvasx(0)
        screen_y = cy - self.app.board.canvasy(0)
        self.assertGreater(screen_x, 0)
        self.assertLess(screen_x, self.app.board.winfo_width())
        self.assertGreater(screen_y, 0)
        self.assertLess(screen_y, self.app.board.winfo_height())
        self.assertEqual(len(self.app.board.find_withtag("visible")), 9)

    def test_success_reveals_initial_key_and_hidden_trap_without_changing_records(self):
        self.app.set_zoom(2.0)
        self.finish_manual_fixture()
        self.assertTrue(self.app.show_full_map.get())
        self.assertEqual(self.app.board.zoom, 1.0)
        self.assertEqual(len(self.app.board.find_withtag("visible")), 9 * 7)
        self.assertEqual(self.app.board.find_withtag("unknown"), ())
        self.assertTrue(self.app.board.find_withtag("icon:K"))
        self.assertTrue(self.app.board.find_withtag("icon:T"))
        self.assertIn("전체 지도", self.app.board_title.get())
        observation = self.app.game.observation
        remembered = dict(self.app.game.memory.known_tiles)
        saved = self.app.game.last_saved
        before = saved.json_path.read_bytes()
        self.app.reveal_toggle.invoke()
        self.assertEqual(len(self.app.board.find_withtag("visible")), len(observation.visible))
        self.app.reveal_toggle.invoke()
        self.assertEqual(len(self.app.board.find_withtag("visible")), 9 * 7)
        self.assertEqual(self.app.game.memory.known_tiles, remembered)
        self.assertEqual(self.app.game.observation, observation)
        self.assertEqual(saved.json_path.read_bytes(), before)
        data = json.loads(before)
        self.assertEqual(len(data["final_observation"]["visible"]), 9)
        self.assertEqual(data["final_memory"]["known_grid"][3][5], "?")
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)

    def test_full_map_is_rejected_before_escape_and_after_action_limit(self):
        with self.assertRaises(RuntimeError):
            self.app.game.get_completed_map()
        observation = self.app.game.observation
        with self.assertRaises(ValueError):
            self.app.board.render(observation, CompletedMap(observation.shape, observation.visible))
        self.app.show_full_map.set(True)
        self.app.refresh()
        self.assert_local_view()
        self.assertTrue(self.app.reveal_toggle.instate(["disabled"]))
        self.app.game.max_actions = 1
        self.app.select_controller("bfs")
        self.app.reveal_toggle.invoke()
        self.assertTrue(self.app.show_memory_map.get())
        self.app.step()
        self.assertTrue(self.app.game.finished)
        self.assertFalse(self.app.game.observation.done)
        with self.assertRaises(RuntimeError):
            self.app.game.get_completed_map()
        self.assert_local_view()
        self.assertTrue(self.app.reveal_toggle.instate(["disabled"]))

    def test_completed_path_shows_actual_moves_and_breaks_at_trap_return(self):
        for action in (Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT):
            self.app.move(action)
        self.assertEqual(self.app.game.observation.trap_hits, 1)
        self.assertEqual(self.app.fields["position"].get(), "(1, 1)")
        self.assertEqual(self.app.fields["key"].get(), "보유")
        self.assertEqual(self.app.fields["goal"].get(), "출구 (1, 4)")
        self.assertIn("열쇠도 계속 보유", self.app.message.get())
        self.assertTrue(self.app.path_toggle.instate(["disabled"]))
        board = self.app.board
        self.assertFalse(board.find_withtag("travel-path"))

        self.finish_manual_fixture()
        center = board.geometry_info.center
        segments = {}
        for line in board.find_withtag("travel-path"):
            x1, y1, x2, y2 = board.coords(line)
            segments[frozenset(((x1, y1), (x2, y2)))] = board.itemcget(line, "fill")
        self.assertEqual(len(segments), 9)
        self.assertEqual(segments[frozenset((center((1, 1)), center((2, 1))))], COLORS["repeated_path"])
        self.assertEqual(segments[frozenset((center((3, 4)), center((3, 5))))], COLORS["path"])
        self.assertNotIn(frozenset((center((3, 5)), center((1, 1)))), segments)

        self.app.path_toggle.invoke()
        self.assertFalse(board.find_withtag("travel-path"))
        self.app.path_toggle.invoke()
        self.assertTrue(board.find_withtag("travel-path"))
        self.app.reveal_toggle.invoke()
        self.assertTrue(self.app.path_toggle.instate(["disabled"]))
        self.assertFalse(board.find_withtag("travel-path"))
        self.app.reveal_toggle.invoke()
        self.assertTrue(board.find_withtag("travel-path"))

        self.app.set_zoom(2.0)
        self.root.geometry("1100x740")
        self.root.update()
        self.assertEqual(len(board.find_withtag("travel-path")), 9)
        self.app.restart()
        self.assertTrue(self.app.path_toggle.instate(["disabled"]))
        self.assertFalse(board.find_withtag("travel-path"))

    def test_restart_controller_and_map_changes_clear_the_completed_view(self):
        self.finish_manual_fixture()
        self.app.restart()
        self.assert_local_view()
        self.finish_manual_fixture()
        self.app.select_controller("bfs")
        self.assert_local_view()
        for _ in range(150):
            if self.app.game.finished:
                break
            self.app.step()
        self.assertTrue(self.app.show_full_map.get())
        self.app.select_map("hard21")
        self.assert_local_view()
        self.assertEqual(self.app.game.observation.shape, (21, 21))

    def test_completed_view_survives_zoom_and_resize(self):
        self.finish_manual_fixture()
        self.root.geometry("1100x740")
        self.root.update()
        self.app.set_zoom(2.0)
        self.root.update()
        self.assertEqual(len(self.app.board.find_withtag("visible")), 9 * 7)
        self.assertIsNotNone(self.app.board.completed_map)
        self.app.set_zoom(1.0)
        geometry = self.app.board.geometry_info
        self.assertLessEqual(geometry.x + geometry.columns * geometry.cell_size, self.app.board.winfo_width())
        self.assertLessEqual(geometry.y + geometry.rows * geometry.cell_size, self.app.board.winfo_height())

    def test_default_enlargement_and_preferred_play_zoom_return_after_review(self):
        self.assertEqual(self.app.board.zoom, 1.5)
        self.assertEqual(self.app.zoom_text.get(), "150%")
        self.finish_manual_fixture()
        self.assertEqual(self.app.board.zoom, 1.0)
        self.app.set_zoom(2.0)  # Zooming the review does not change the play preference.
        self.app.restart()
        self.assertEqual(self.app.board.zoom, 1.5)
        self.app.set_zoom(1.0)
        self.finish_manual_fixture()
        self.app.restart()
        self.assertEqual(self.app.board.zoom, 1.0)
        self.assertEqual(self.app.zoom_text.get(), "화면 맞춤")


if __name__ == "__main__":
    unittest.main()
