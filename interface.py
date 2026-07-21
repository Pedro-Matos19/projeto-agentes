from __future__ import annotations

import math
import os
import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from aestrela import AStarPlanner
from algoritmogenetico import GeneticPlanner
from ambiente import generate_scenario
from metricas import compare_plans
from modelos import AgentRuntimeState, PlannerResult, Scenario
from resultados import save_results


BG = (13, 18, 22)
MAP_BG = (17, 23, 27)
PANEL_BG = (20, 27, 32)
DIVIDER = (48, 61, 68)
TEXT = (229, 235, 237)
MUTED = (132, 148, 156)
FAINT = (73, 87, 94)
WALL = (174, 190, 196)
DOCK_BG = (8, 54, 73)
DOCK_BLUE = (20, 181, 226)
PRODUCT = (244, 166, 57)
ASTAR = (246, 103, 91)
GENETIC = (174, 112, 230)
SUCCESS = (95, 205, 145)
ERROR = (244, 105, 105)
AGENTS = (("astar", ASTAR), ("genetic", GENETIC))
STATE_LABELS = {
    "READY": "Cenário pronto",
    "PLANNING": "Planejando",
    "COUNTDOWN": "Sincronizando",
    "RUNNING": "Em operação",
    "PAUSED": "Pausado",
    "FINISHED": "Operação concluída",
    "ERROR": "Falha no planejamento",
}
PRIMARY_LABELS = {
    "READY": "Iniciar comparação",
    "PLANNING": "Planejando rotas…",
    "COUNTDOWN": "Preparando agentes…",
    "RUNNING": "Pausar simulação",
    "PAUSED": "Continuar simulação",
    "FINISHED": "Executar novamente",
    "ERROR": "Tentar novamente",
}
BUSY_STATES = {"PLANNING", "COUNTDOWN"}
RESULTS_FILE = Path(__file__).with_name("resultados_armazem.csv")


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: str
    enabled: bool = True
    primary: bool = False


class MapViewport:
    def __init__(self, scenario: Scenario, viewport: pygame.Rect) -> None:
        self.scenario = scenario
        self.viewport = viewport.copy()
        self.cell_size = 8
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.static_surface = pygame.Surface((1, 1))
        self.fit()

    @property
    def map_width(self) -> int:
        return self.scenario.maze.total_cols * self.cell_size

    @property
    def map_height(self) -> int:
        return self.scenario.maze.rows * self.cell_size

    @property
    def origin(self) -> tuple[float, float]:
        return self.viewport.x + self.pan_x, self.viewport.y + self.pan_y

    def set_scenario(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.fit()

    def set_viewport(self, viewport: pygame.Rect) -> None:
        self.viewport = viewport.copy()
        self.fit()

    def fit(self) -> None:
        horizontal = max(2, int(self.viewport.width / self.scenario.maze.total_cols))
        vertical = max(2, int(self.viewport.height / self.scenario.maze.rows))
        self.cell_size = max(2, min(horizontal, vertical))
        self.pan_x = (self.viewport.width - self.map_width) / 2
        self.pan_y = (self.viewport.height - self.map_height) / 2
        self._rebuild_static()

    def zoom(self, direction: int, cursor: tuple[int, int]) -> None:
        if not self.viewport.collidepoint(cursor):
            return
        old_size = self.cell_size
        new_size = max(2, min(40, old_size + (2 if direction > 0 else -2)))
        if new_size == old_size:
            return
        origin_x, origin_y = self.origin
        world_col = (cursor[0] - origin_x) / old_size
        world_row = (cursor[1] - origin_y) / old_size
        self.cell_size = new_size
        self.pan_x = cursor[0] - self.viewport.x - world_col * new_size
        self.pan_y = cursor[1] - self.viewport.y - world_row * new_size
        self._clamp_pan()
        self._rebuild_static()

    def pan(self, dx: float, dy: float) -> None:
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        margin = 36
        if self.map_width <= self.viewport.width:
            self.pan_x = (self.viewport.width - self.map_width) / 2
        else:
            self.pan_x = min(margin, max(self.viewport.width - self.map_width - margin, self.pan_x))
        if self.map_height <= self.viewport.height:
            self.pan_y = (self.viewport.height - self.map_height) / 2
        else:
            self.pan_y = min(margin, max(self.viewport.height - self.map_height - margin, self.pan_y))

    def cell_center(self, cell: tuple[float, float]) -> tuple[int, int]:
        row, col = cell
        origin_x, origin_y = self.origin
        return (
            round(origin_x + (col + 0.5) * self.cell_size),
            round(origin_y + (row + 0.5) * self.cell_size),
        )

    def is_visible(self, point: tuple[int, int], margin: int = 20) -> bool:
        return self.viewport.inflate(margin * 2, margin * 2).collidepoint(point)

    def draw(self, destination: pygame.Surface) -> None:
        old_clip = destination.get_clip()
        destination.set_clip(self.viewport)
        destination.blit(self.static_surface, (round(self.origin[0]), round(self.origin[1])))
        destination.set_clip(old_clip)

    def _rebuild_static(self) -> None:
        width = max(1, self.map_width)
        height = max(1, self.map_height)
        surface = pygame.Surface((width, height))
        surface.fill(MAP_BG)
        dock_width_px = self.scenario.maze.dock_width * self.cell_size
        pygame.draw.rect(surface, DOCK_BG, (0, 0, dock_width_px, height))

        stripe = max(8, self.cell_size * 3)
        surface.set_clip(pygame.Rect(0, 0, dock_width_px, height))
        for offset in range(-height, dock_width_px + height, stripe):
            pygame.draw.line(
                surface,
                (10, 69, 91),
                (offset, height),
                (offset + height, 0),
                max(1, self.cell_size // 7),
            )
        surface.set_clip(None)

        maze = self.scenario.maze
        line_width = 2 if self.cell_size >= 9 else 1
        for row in range(maze.rows):
            for col in range(maze.total_cols):
                cell = (row, col)
                x = col * self.cell_size
                y = row * self.cell_size
                north = (row - 1, col)
                west = (row, col - 1)
                if row == 0 or not maze.connected(cell, north):
                    pygame.draw.line(surface, WALL, (x, y), (x + self.cell_size, y), line_width)
                if col == 0 or not maze.connected(cell, west):
                    pygame.draw.line(surface, WALL, (x, y), (x, y + self.cell_size), line_width)
                if row == maze.rows - 1:
                    pygame.draw.line(
                        surface,
                        WALL,
                        (x, y + self.cell_size - 1),
                        (x + self.cell_size, y + self.cell_size - 1),
                        line_width,
                    )
                if col == maze.total_cols - 1:
                    pygame.draw.line(
                        surface,
                        WALL,
                        (x + self.cell_size - 1, y),
                        (x + self.cell_size - 1, y + self.cell_size),
                        line_width,
                    )

        for row in range(maze.rows):
            dock_cell = (row, maze.dock_width - 1)
            warehouse_cell = (row, maze.dock_width)
            if not maze.connected(dock_cell, warehouse_cell):
                pygame.draw.line(
                    surface,
                    DOCK_BLUE,
                    (dock_width_px - 1, row * self.cell_size),
                    (dock_width_px - 1, (row + 1) * self.cell_size),
                    max(1, line_width),
                )
        self.static_surface = surface


class WarehouseApp:
    BASE_STEPS_PER_SECOND = 35.0
    SPEEDS = (0.5, 1.0, 2.0, 4.0)

    def __init__(
        self,
        rows: int = 35,
        cols: int = 55,
        product_count: int = 8,
        seed: int = 20260721,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Warehouse Route Lab — Agentes Inteligentes")
        self.screen = pygame.display.set_mode((1480, 900), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "eyebrow": pygame.font.SysFont("Segoe UI", 12, bold=True),
            "small": pygame.font.SysFont("Segoe UI", 13),
            "body": pygame.font.SysFont("Segoe UI", 15),
            "body_bold": pygame.font.SysFont("Segoe UI", 15, bold=True),
            "title": pygame.font.SysFont("Segoe UI", 27, bold=True),
            "metric": pygame.font.SysFont("Segoe UI", 22, bold=True),
            "countdown": pygame.font.SysFont("Segoe UI", 76, bold=True),
        }
        self.rows = rows
        self.cols = cols
        self.product_count = max(4, min(8, product_count))
        self.seed = seed
        self.state = "READY"
        self.error_message = ""
        self.save_error = ""
        self.plans: dict[str, PlannerResult] = {}
        self.runtimes: dict[str, AgentRuntimeState] = {}
        self.astar_progress: dict[str, float | int | str] = {}
        self.ga_progress: dict[str, float | int | str] = {}
        self.ga_history: list[int] = []
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.countdown = 0.0
        self.step_phase = 0.0
        self.simulated_seconds = 0.0
        self.speed_index = 1
        self.results_saved = False
        self.dragging = False
        self.buttons: list[Button] = []
        self.scenario = self._create_scenario()
        self.map_rect, self.sidebar_rect = self._layout_rects()
        self.viewport = MapViewport(self.scenario, self.map_rect)
        self._refresh_buttons()

    def _create_scenario(self) -> Scenario:
        return generate_scenario(self.rows, self.cols, self.product_count, self.seed)

    def _layout_rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        width, height = self.screen.get_size()
        sidebar_width = 350 if width >= 1280 else 320
        sidebar = pygame.Rect(width - sidebar_width, 0, sidebar_width, height)
        map_rect = pygame.Rect(24, 88, width - sidebar_width - 48, height - 112)
        return map_rect, sidebar

    def _reset_for_scenario(self) -> None:
        self._clear_run()
        self.state = "READY"
        self.scenario = self._create_scenario()
        self.viewport.set_scenario(self.scenario)

    def _clear_run(self) -> None:
        self.plans.clear()
        self.runtimes.clear()
        self.astar_progress.clear()
        self.ga_progress.clear()
        self.ga_history.clear()
        self.error_message = ""
        self.save_error = ""
        self.step_phase = 0.0
        self.simulated_seconds = 0.0
        self.results_saved = False

    def _new_scenario(self) -> None:
        self.seed = random.SystemRandom().randrange(1, 1_000_000_000)
        self._reset_for_scenario()

    def _restart_same(self) -> None:
        self._reset_for_scenario()
        self._start_planning()

    def _start_planning(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self._clear_run()
        self.state = "PLANNING"
        self.messages = queue.Queue()
        scenario = self.scenario
        self.worker = threading.Thread(
            target=self._planning_worker,
            args=(scenario,),
            daemon=True,
            name="warehouse-planners",
        )
        self.worker.start()

    def _planning_worker(self, scenario: Scenario) -> None:
        try:
            astar_result = AStarPlanner().plan(
                scenario,
                lambda progress: self.messages.put(("astar_progress", progress)),
            )
            self.messages.put(("astar_done", astar_result))
            genetic_result = GeneticPlanner().plan(
                scenario,
                lambda progress: self.messages.put(("ga_progress", progress)),
            )
            self.messages.put(("planning_done", (astar_result, genetic_result)))
        except Exception as exc:  # a mensagem aparece na própria janela
            self.messages.put(("error", str(exc)))

    def _process_messages(self) -> None:
        while True:
            try:
                event, payload = self.messages.get_nowait()
            except queue.Empty:
                return

            if event == "astar_progress":
                self.astar_progress = dict(payload)
            elif event == "astar_done":
                self.plans["astar"] = payload
            elif event == "ga_progress":
                self.ga_progress = dict(payload)
                distance = int(self.ga_progress.get("melhor_distancia", 0))
                if distance:
                    self.ga_history.append(distance)
            elif event == "planning_done":
                astar_result, genetic_result = payload
                self.plans = {"astar": astar_result, "genetic": genetic_result}
                if not astar_result.success or not genetic_result.success:
                    self.state = "ERROR"
                    self.error_message = "Um dos agentes não conseguiu construir uma rota completa."
                    continue
                self.runtimes = {
                    "astar": AgentRuntimeState(astar_result),
                    "genetic": AgentRuntimeState(genetic_result),
                }
                self.countdown = 3.6
                self.simulated_seconds = 0.0
                self.step_phase = 0.0
                self.state = "COUNTDOWN"
            elif event == "error":
                self.state = "ERROR"
                self.error_message = str(payload)

    def _update(self, dt: float) -> None:
        if self.state == "COUNTDOWN":
            self.countdown -= dt
            if self.countdown <= 0:
                for runtime in self.runtimes.values():
                    runtime.begin()
                self.state = "RUNNING"
        elif self.state == "RUNNING":
            speed = self.SPEEDS[self.speed_index]
            self.simulated_seconds += dt * speed
            self.step_phase += dt * self.BASE_STEPS_PER_SECOND * speed
            while self.step_phase >= 1:
                for runtime in self.runtimes.values():
                    runtime.advance(self.simulated_seconds)
                self.step_phase -= 1
            if self.runtimes and all(runtime.finished for runtime in self.runtimes.values()):
                self.state = "FINISHED"
                if not self.results_saved:
                    try:
                        save_results(
                            self.scenario,
                            self.runtimes,
                            RESULTS_FILE,
                        )
                    except OSError as exc:
                        self.save_error = f"Não foi possível salvar o CSV: {exc}"
                    self.results_saved = True

    def _primary_action(self) -> None:
        if self.state in {"READY", "FINISHED", "ERROR"}:
            self._start_planning()
        elif self.state == "RUNNING":
            self.state = "PAUSED"
        elif self.state == "PAUSED":
            self.state = "RUNNING"

    def _perform_action(self, action: str) -> None:
        if action == "primary":
            self._primary_action()
        elif action == "restart":
            self._restart_same()
        elif action == "new":
            self._new_scenario()
        elif action == "speed":
            self.speed_index = (self.speed_index + 1) % len(self.SPEEDS)
        elif action == "fit":
            self.viewport.fit()
        elif action == "size":
            self.rows, self.cols = ((100, 100) if self.rows != 100 else (35, 55))
            self._reset_for_scenario()
        elif action == "products_down" and self.product_count > 4:
            self.product_count -= 1
            self._reset_for_scenario()
        elif action == "products_up" and self.product_count < 8:
            self.product_count += 1
            self._reset_for_scenario()

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            size = (max(1100, event.w), max(720, event.h))
            self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            self.map_rect, self.sidebar_rect = self._layout_rects()
            self.viewport.set_viewport(self.map_rect)
        elif event.type == pygame.MOUSEWHEEL:
            self.viewport.zoom(event.y, pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                for button in self.buttons:
                    if button.enabled and button.rect.collidepoint(event.pos):
                        self._perform_action(button.action)
                        break
            elif event.button in (2, 3) and self.map_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.viewport.pan(event.rel[0], event.rel[1])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self._primary_action()
            elif event.key == pygame.K_f:
                self.viewport.fit()
            elif event.key == pygame.K_r and self.state not in BUSY_STATES:
                self._restart_same()
            elif event.key == pygame.K_n and self.state not in BUSY_STATES:
                self._new_scenario()
        return True

    def _refresh_buttons(self) -> None:
        x = self.sidebar_rect.x + 24
        width = self.sidebar_rect.width - 48
        bottom = self.sidebar_rect.bottom
        configurable = self.state in {"READY", "FINISHED", "ERROR"}
        busy = self.state in BUSY_STATES
        primary_label = PRIMARY_LABELS[self.state]

        self.buttons = [
            Button(pygame.Rect(x, 120, 116, 34), f"{self.rows} × {self.cols}", "size", configurable),
            Button(pygame.Rect(x + 126, 120, 34, 34), "−", "products_down", configurable and self.product_count > 4),
            Button(pygame.Rect(x + 166, 120, width - 206, 34), f"{self.product_count} itens", "noop", False),
            Button(pygame.Rect(x + width - 34, 120, 34, 34), "+", "products_up", configurable and self.product_count < 8),
            Button(pygame.Rect(x, bottom - 188, width, 44), primary_label, "primary", not busy, True),
            Button(pygame.Rect(x, bottom - 134, (width - 10) // 2, 38), "Reiniciar", "restart", not busy),
            Button(pygame.Rect(x + (width + 10) // 2, bottom - 134, (width - 10) // 2, 38), "Novo cenário", "new", not busy),
            Button(pygame.Rect(x, bottom - 86, (width - 10) // 2, 36), f"Velocidade {self.SPEEDS[self.speed_index]:g}×", "speed"),
            Button(pygame.Rect(x + (width + 10) // 2, bottom - 86, (width - 10) // 2, 36), "Enquadrar", "fit"),
        ]

    def _text(
        self,
        text: object,
        position: tuple[int, int],
        font: str = "body",
        color: tuple[int, int, int] = TEXT,
    ) -> pygame.Rect:
        surface = self.fonts[font].render(str(text), True, color)
        return self.screen.blit(surface, position)

    def _truncate(self, text: str, max_width: int, font: str = "small") -> str:
        if self.fonts[font].size(text)[0] <= max_width:
            return text
        result = text
        while result and self.fonts[font].size(result + "…")[0] > max_width:
            result = result[:-1]
        return result + "…"

    def _draw_header(self) -> None:
        self._text("WAREHOUSE // ROUTE LAB", (24, 18), "eyebrow", DOCK_BLUE)
        self._text("Coleta autônoma", (24, 37), "title")
        label = STATE_LABELS[self.state]
        label_width = self.fonts["small"].size(label)[0]
        x = self.map_rect.right - label_width - 18
        pygame.draw.circle(self.screen, SUCCESS if self.state == "FINISHED" else DOCK_BLUE, (x, 52), 4)
        self._text(label, (x + 10, 43), "small", MUTED)

    def _draw_trails(self) -> None:
        if not self.runtimes:
            return
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for key, color in AGENTS:
            runtime = self.runtimes[key]
            points = [self.viewport.cell_center(cell) for cell in runtime.trail]
            for index in range(1, len(points)):
                alpha = int(25 + 105 * index / max(1, len(points) - 1))
                pygame.draw.line(overlay, (*color, alpha), points[index - 1], points[index], 3)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.map_rect)
        self.screen.blit(overlay, (0, 0))
        self.screen.set_clip(old_clip)

    def _draw_docks(self) -> None:
        size = max(5, min(16, round(self.viewport.cell_size * 0.72)))
        for slot in self.scenario.dock_slots:
            center = self.viewport.cell_center(slot.position)
            if not self.viewport.is_visible(center):
                continue
            rect = pygame.Rect(center[0] - size // 2, center[1] - size // 2, size, size)
            pygame.draw.rect(self.screen, (15, 41, 51), rect)
            pygame.draw.rect(self.screen, DOCK_BLUE, rect, max(1, size // 6), border_radius=2)
            if "astar" in self.runtimes and slot.id in self.runtimes["astar"].delivered_slots:
                half = pygame.Rect(rect.x + 2, rect.y + 2, max(1, rect.width // 2 - 2), rect.height - 4)
                pygame.draw.rect(self.screen, ASTAR, half)
            if "genetic" in self.runtimes and slot.id in self.runtimes["genetic"].delivered_slots:
                half = pygame.Rect(rect.centerx, rect.y + 2, max(1, rect.width // 2 - 2), rect.height - 4)
                pygame.draw.rect(self.screen, GENETIC, half)

    def _draw_products(self) -> None:
        radius = max(4, min(11, round(self.viewport.cell_size * 0.34)))
        for product in self.scenario.products:
            center = self.viewport.cell_center(product.position)
            if not self.viewport.is_visible(center):
                continue
            astar_status = self.runtimes.get("astar").product_status.get(product.id) if "astar" in self.runtimes else "pending"
            genetic_status = self.runtimes.get("genetic").product_status.get(product.id) if "genetic" in self.runtimes else "pending"
            completed = astar_status == genetic_status == "delivered"
            box_color = FAINT if completed else PRODUCT
            rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, (28, 32, 33), rect, border_radius=3)
            pygame.draw.rect(self.screen, box_color, rect, 2, border_radius=3)
            pygame.draw.line(self.screen, box_color, rect.midtop, rect.center, 1)
            pygame.draw.line(self.screen, box_color, rect.center, rect.midleft, 1)
            dot_radius = max(2, radius // 4)
            pygame.draw.circle(self.screen, ASTAR if astar_status == "pending" else FAINT, (center[0] - radius // 2, center[1] + radius + 4), dot_radius)
            pygame.draw.circle(self.screen, GENETIC if genetic_status == "pending" else FAINT, (center[0] + radius // 2, center[1] + radius + 4), dot_radius)
            if self.viewport.cell_size >= 16:
                label = self.fonts["eyebrow"].render(str(product.id), True, TEXT)
                self.screen.blit(label, label.get_rect(center=center))

    def _draw_target_pulses(self) -> None:
        if self.state not in {"RUNNING", "PAUSED"}:
            return
        products = {product.id: product.position for product in self.scenario.products}
        pulse = 4 + int(3 * (1 + math.sin(pygame.time.get_ticks() / 220)) / 2)
        for key, color in AGENTS:
            runtime = self.runtimes[key]
            if runtime.carrying is None and runtime.target_product in products:
                center = self.viewport.cell_center(products[runtime.target_product])
                pygame.draw.circle(self.screen, color, center, max(8, self.viewport.cell_size // 2) + pulse, 1)

    def _draw_agents(self) -> None:
        if not self.runtimes:
            start = self.viewport.cell_center(self.scenario.start)
            pygame.draw.circle(self.screen, DOCK_BLUE, start, max(5, self.viewport.cell_size // 3), 2)
            return

        interpolation = self.step_phase if self.state in {"RUNNING", "PAUSED"} else 0.0
        for key, color, offset in (("astar", ASTAR, -0.17), ("genetic", GENETIC, 0.17)):
            runtime = self.runtimes[key]
            current = runtime.position
            upcoming = runtime.next_position
            row = current[0] + (upcoming[0] - current[0]) * interpolation
            col = current[1] + (upcoming[1] - current[1]) * interpolation + offset
            center = self.viewport.cell_center((row, col))
            radius = max(4, min(10, round(self.viewport.cell_size * 0.31)))
            pygame.draw.circle(self.screen, (4, 8, 10), (center[0] + 2, center[1] + 3), radius + 2)
            pygame.draw.circle(self.screen, color, center, radius + 2)
            pygame.draw.circle(self.screen, (244, 248, 249), center, max(2, radius // 2))
            if runtime.carrying is not None:
                cargo = (center[0] + radius, center[1] - radius)
                pygame.draw.rect(self.screen, PRODUCT, (cargo[0] - 3, cargo[1] - 3, 7, 7), border_radius=1)

    def _draw_map_content(self) -> None:
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.map_rect)
        self._draw_trails()
        self._draw_docks()
        self._draw_products()
        self._draw_target_pulses()
        self._draw_agents()
        self.screen.set_clip(old_clip)

    def _draw_agent_summary(self, key: str, title: str, color: tuple[int, int, int], y: int) -> None:
        x = self.sidebar_rect.x + 24
        width = self.sidebar_rect.width - 48
        pygame.draw.rect(self.screen, color, (x, y, 3, 88), border_radius=2)
        self._text(title, (x + 13, y - 2), "body_bold", color)
        runtime = self.runtimes.get(key)
        plan = self.plans.get(key)
        if runtime:
            status = runtime.status
            delivered = runtime.delivered_count
            steps = runtime.steps
        elif plan:
            status = "Plano concluído"
            delivered = 0
            steps = int(plan.metrics.get("passos", 0))
        elif key == "astar" and self.astar_progress:
            expanded = int(self.astar_progress.get("estados_expandidos", 0))
            status = f"Explorando {expanded:,} estados".replace(",", ".")
            delivered = 0
            steps = 0
        elif key == "genetic" and self.ga_progress:
            generation = int(self.ga_progress.get("geracao", 0))
            status = f"Evoluindo geração {generation}"
            delivered = 0
            steps = int(self.ga_progress.get("melhor_distancia", 0))
        else:
            status = "Aguardando planejamento"
            delivered = 0
            steps = 0
        self._text(self._truncate(status, width - 18), (x + 13, y + 23), "small", MUTED)
        self._text(f"{delivered}/{self.product_count}", (x + 13, y + 51), "metric")
        self._text("entregas", (x + 13, y + 73), "eyebrow", MUTED)
        self._text(str(steps), (x + 103, y + 51), "metric")
        self._text("passos", (x + 103, y + 73), "eyebrow", MUTED)
        planning = f"{plan.planning_ms:.0f} ms" if plan else "—"
        self._text(planning, (x + 193, y + 51), "metric")
        self._text("planejamento", (x + 193, y + 73), "eyebrow", MUTED)

    def _draw_ga_chart(self, rect: pygame.Rect) -> None:
        pygame.draw.line(self.screen, DIVIDER, rect.bottomleft, rect.bottomright)
        if len(self.ga_history) < 2:
            self._text("A evolução aparecerá durante o planejamento.", (rect.x, rect.y + 12), "small", MUTED)
            return
        values = self.ga_history[-80:]
        minimum, maximum = min(values), max(values)
        span = max(1, maximum - minimum)
        points = []
        for index, value in enumerate(values):
            x = rect.x + index * rect.width / max(1, len(values) - 1)
            y = rect.bottom - 5 - (maximum - value) * (rect.height - 10) / span
            points.append((round(x), round(y)))
        pygame.draw.lines(self.screen, GENETIC, False, points, 2)

    def _draw_sidebar(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, self.sidebar_rect)
        pygame.draw.line(self.screen, DIVIDER, self.sidebar_rect.topleft, self.sidebar_rect.bottomleft)
        x = self.sidebar_rect.x + 24
        width = self.sidebar_rect.width - 48
        self._text("PAINEL OPERACIONAL", (x, 22), "eyebrow", MUTED)
        self._text("Comparação de agentes", (x, 43), "title")
        self._text(f"Seed {self.seed}  ·  {self.scenario.rows}×{self.scenario.cols}", (x, 82), "small", MUTED)
        self._text("CENÁRIO", (x, 101), "eyebrow", DOCK_BLUE)

        pygame.draw.line(self.screen, DIVIDER, (x, 170), (x + width, 170))
        self._text("EXECUÇÃO", (x, 187), "eyebrow", MUTED)
        self._text(self._state_description(), (x, 207), "body", TEXT if self.state != "ERROR" else ERROR)

        self._draw_agent_summary("astar", "A* · busca global", ASTAR, 244)
        self._draw_agent_summary("genetic", "Genético · ordem global", GENETIC, 354)

        generation = int(self.ga_progress.get("geracao", 0))
        generation_max = int(self.ga_progress.get("geracoes_maximas", 300))
        stagnation = int(self.ga_progress.get("sem_melhoria", 0))
        self._text("EVOLUÇÃO GENÉTICA", (x, 463), "eyebrow", MUTED)
        self._text(f"Geração {generation}/{generation_max}  ·  sem melhora {stagnation}", (x, 481), "small", MUTED)
        self._draw_ga_chart(pygame.Rect(x, 503, width, 40))

        if self.sidebar_rect.height >= 820:
            self._text("LEGENDA", (x, 565), "eyebrow", MUTED)
            legends = ((ASTAR, "A*"), (GENETIC, "Genético"), (PRODUCT, "Produto pendente"), (DOCK_BLUE, "Docas"))
            for index, (color, label) in enumerate(legends):
                line_x = x + (index % 2) * 145
                line_y = 588 + (index // 2) * 25
                pygame.draw.circle(self.screen, color, (line_x + 5, line_y + 7), 5)
                self._text(label, (line_x + 17, line_y), "small", MUTED)

        for button in self.buttons:
            self._draw_button(button)

    def _state_description(self) -> str:
        if self.state == "READY":
            return "Produtos alocados; agentes na origem."
        if self.state == "PLANNING":
            return "Calculando as duas políticas de coleta."
        if self.state == "COUNTDOWN":
            return "Planos prontos; largada sincronizada."
        if self.state == "RUNNING":
            return f"Relógio simulado {self.simulated_seconds:05.1f} s"
        if self.state == "PAUSED":
            return f"Pausado em {self.simulated_seconds:05.1f} s"
        if self.state == "FINISHED":
            return f"Concluído em {self.simulated_seconds:05.1f} s"
        return self._truncate(self.error_message or "Falha desconhecida.", self.sidebar_rect.width - 48, "body")

    def _draw_button(self, button: Button) -> None:
        hovered = button.enabled and button.rect.collidepoint(pygame.mouse.get_pos())
        if button.primary:
            fill = (13, 133, 171) if button.enabled else (39, 58, 66)
            if hovered:
                fill = (19, 158, 200)
            border = fill
            color = TEXT if button.enabled else MUTED
        else:
            fill = (29, 39, 45) if button.enabled else (24, 31, 35)
            if hovered:
                fill = (38, 51, 58)
            border = (59, 75, 83)
            color = TEXT if button.enabled else FAINT
        pygame.draw.rect(self.screen, fill, button.rect, border_radius=5)
        pygame.draw.rect(self.screen, border, button.rect, 1, border_radius=5)
        label = self.fonts["body_bold" if button.primary else "small"].render(button.label, True, color)
        self.screen.blit(label, label.get_rect(center=button.rect.center))

    def _draw_center_overlay(self) -> None:
        if self.state not in {"READY", "PLANNING", "COUNTDOWN", "ERROR"}:
            return
        center = self.map_rect.center
        overlay = pygame.Surface((430, 132), pygame.SRCALPHA)
        overlay.fill((10, 15, 18, 225))
        self.screen.blit(overlay, overlay.get_rect(center=center))
        if self.state == "COUNTDOWN":
            number = max(1, math.ceil(self.countdown))
            rendered = self.fonts["countdown"].render(str(number), True, DOCK_BLUE)
            self.screen.blit(rendered, rendered.get_rect(center=(center[0], center[1] - 8)))
            label = "AGENTES SINCRONIZADOS"
        elif self.state == "PLANNING":
            label = "PLANEJANDO ROTAS"
            self._text("A interface permanece ativa durante a evolução.", (center[0] - 158, center[1] + 12), "small", MUTED)
        elif self.state == "ERROR":
            label = "PLANEJAMENTO INTERROMPIDO"
            self._text(self._truncate(self.error_message, 370), (center[0] - 185, center[1] + 12), "small", ERROR)
        else:
            label = "CENÁRIO PRONTO"
            self._text("Inicie para calcular e comparar as duas rotas.", (center[0] - 164, center[1] + 12), "small", MUTED)
        rendered = self.fonts["body_bold"].render(label, True, TEXT)
        self.screen.blit(rendered, rendered.get_rect(center=(center[0], center[1] - 25 if self.state != "COUNTDOWN" else center[1] + 45)))

    def _draw_final_summary(self) -> None:
        if self.state != "FINISHED" or not self.runtimes:
            return
        astar = self.runtimes["astar"]
        genetic = self.runtimes["genetic"]
        comparison = compare_plans(astar.plan, genetic.plan)
        if astar.steps == genetic.steps:
            winner = "Genético também alcançou a rota ótima"
        else:
            winner = "A* encontrou a operação de menor custo"

        rect = pygame.Rect(
            self.map_rect.x + 18,
            self.map_rect.bottom - 176,
            min(850, self.map_rect.width - 36),
            156,
        )
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((8, 13, 16, 232))
        self.screen.blit(overlay, rect)
        self._text("RESULTADO FINAL", (rect.x + 18, rect.y + 13), "eyebrow", SUCCESS)
        self._text(winner, (rect.x + 18, rect.y + 32), "body_bold")

        available = rect.width - 36
        lines = (
            f"ROTAS  A* {astar.steps}  ·  Genético {genetic.steps}  ·  "
            f"gap {comparison.optimality_gap_percent:.2f}%  ·  "
            f"Genético: ótimo - {'sim' if comparison.ga_reached_optimum else 'não'}",
            f"PLANEJAMENTO  A* {self._format_duration(astar.plan.planning_ms)}  ·  "
            f"Genético {self._format_duration(genetic.plan.planning_ms)}  ·  "
            f"{comparison.faster_planner} {comparison.planning_speed_ratio:.1f}× mais rápido",
            f"TRAJETO  curvas A* {comparison.astar.turns} · G {comparison.genetic.turns}  ·  "
            f"revisitas A* {comparison.astar.revisit_percent:.1f}% · "
            f"G {comparison.genetic.revisit_percent:.1f}%  ·  "
            f"sobreposição {comparison.route_overlap_percent:.1f}%",
            f"DECISÕES  ordem de produtos {comparison.product_order_agreement_percent:.1f}%  ·  "
            f"ordem de docas {comparison.dock_order_agreement_percent:.1f}%  ·  "
            f"passos/entrega A* {comparison.astar.steps_per_delivery:.1f} · "
            f"G {comparison.genetic.steps_per_delivery:.1f}",
            (
                f"ESFORÇO  A* {int(astar.plan.metrics.get('nos_expandidos', 0)):,} estados  ·  "
                f"G {int(genetic.plan.metrics.get('geracoes', 0))} gerações  ·  "
                f"memória A* {self._format_memory(astar.plan.metrics.get('memoria_pico_kib', 0))} · "
                f"G {self._format_memory(genetic.plan.metrics.get('memoria_pico_kib', 0))}"
            ).replace(",", "."),
        )
        for index, line in enumerate(lines):
            color = TEXT if index == 0 else MUTED
            self._text(
                self._truncate(line, available, "small"),
                (rect.x + 18, rect.y + 58 + index * 18),
                "small",
                color,
            )
        if self.save_error:
            self._text(self._truncate(self.save_error, rect.width - 36), (rect.x + 18, rect.bottom + 2), "small", ERROR)

    @staticmethod
    def _format_duration(milliseconds: float) -> str:
        return f"{milliseconds:.0f} ms" if milliseconds < 1000 else f"{milliseconds / 1000:.1f} s"

    @staticmethod
    def _format_memory(value: object) -> str:
        kib = float(value)
        return f"{kib:.0f} KiB" if kib < 1024 else f"{kib / 1024:.1f} MiB"

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_header()
        self.viewport.draw(self.screen)
        pygame.draw.rect(self.screen, DIVIDER, self.map_rect, 1)
        self._draw_map_content()
        self._draw_center_overlay()
        self._draw_final_summary()
        self._draw_sidebar()
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            dt = min(0.05, self.clock.tick(60) / 1000)
            for event in pygame.event.get():
                if not self._handle_event(event):
                    running = False
                    break
            self._process_messages()
            self._update(dt)
            self._refresh_buttons()
            self._draw()
        pygame.quit()
