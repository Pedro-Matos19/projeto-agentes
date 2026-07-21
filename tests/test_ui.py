from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame

    from interface import WarehouseApp
except ImportError:  # permite rodar os testes centrais antes de instalar requirements.txt
    pygame = None
    WarehouseApp = None


@unittest.skipIf(pygame is None, "pygame-ce não está instalado")
class InterfaceTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_planning_animation_and_large_viewport(self) -> None:
        app = WarehouseApp(rows=13, cols=19, product_count=4, seed=91)
        app._start_planning()
        deadline = time.monotonic() + 15
        while app.worker and app.worker.is_alive() and time.monotonic() < deadline:
            app._process_messages()
            time.sleep(0.005)
        app._process_messages()
        self.assertEqual(app.state, "COUNTDOWN")

        app.countdown = 0
        app._update(0.01)
        app.results_saved = True
        for _ in range(5000):
            app._update(0.05)
            if app.state == "FINISHED":
                break
        self.assertEqual(app.state, "FINISHED")
        self.assertTrue(all(runtime.delivered_count == 4 for runtime in app.runtimes.values()))
        app._refresh_buttons()
        app._draw()

        large = WarehouseApp(rows=100, cols=100, product_count=8, seed=91)
        initial_size = large.viewport.cell_size
        large.viewport.zoom(1, large.map_rect.center)
        large.viewport.pan(-50, -40)
        self.assertGreaterEqual(large.viewport.cell_size, initial_size)
        large._draw()


if __name__ == "__main__":
    unittest.main()
