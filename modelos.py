from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ambiente import MazeGrid

"cordenada (linha, coluna)"
Cell = tuple[int, int]

"possíveis tipos de valores que podem aparecer nas estatísticas"
MetricValue = float | int | str | bool

"função para comunicar progresso a interface"
ProgressCallback = Callable[[dict[str, float | int | str]], None]


"produto que será buscado"
@dataclass(frozen=True)
class Product:
    id: int
    position: Cell


"slot disponível na doca"
@dataclass(frozen=True)
class DockSlot:
    id: int
    position: Cell


"agrupa todo o cenário do labirinto"
@dataclass(frozen=True)
class Scenario:
    maze: "MazeGrid"
    start: Cell
    products: tuple[Product, ...]
    dock_slots: tuple[DockSlot, ...]
    seed: int
    ga_seed: int

    @property
    def rows(self) -> int:
        return self.maze.rows

    @property
    def cols(self) -> int:
        return self.maze.warehouse_cols


"representa cada entrega individual"
@dataclass(frozen=True)
class DeliveryPlan:
    product_id: int
    dock_id: int
    pickup_step: int
    delivery_step: int


"representa a respota de um planejador"
@dataclass
class PlannerResult:
    algorithm: str
    route: list[Cell]
    deliveries: list[DeliveryPlan]
    planning_ms: float
    metrics: dict[str, MetricValue]
    success: bool = True


"representa um agente enquando a animação ocorre"
@dataclass
class AgentRuntimeState:

    plan: PlannerResult
    route_index: int = 0
    steps: int = 0
    carrying: int | None = None
    delivered_slots: dict[int, int] = field(default_factory=dict)
    product_status: dict[int, str] = field(default_factory=dict)
    status: str = "Aguardando largada"
    finished_at: float | None = None
    trail: deque[Cell] = field(default_factory=lambda: deque(maxlen=42))
    _events: dict[int, tuple[str, DeliveryPlan]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.product_status = {
            delivery.product_id: "pending" for delivery in self.plan.deliveries
        }
        self.trail.append(self.position)
        for delivery in self.plan.deliveries:
            self._events[delivery.pickup_step] = ("pickup", delivery)
            self._events[delivery.delivery_step] = ("delivery", delivery)

    "as funções abaixo representam as operações que ocorrem durante o trajeto do agente"
    @property
    def position(self) -> Cell:
        return self.plan.route[self.route_index]

    @property
    def next_position(self) -> Cell:
        if self.route_index + 1 < len(self.plan.route):
            return self.plan.route[self.route_index + 1]
        return self.position

    @property
    def delivered_count(self) -> int:
        return len(self.delivered_slots)

    @property
    def finished(self) -> bool:
        return self.finished_at is not None

    @property
    def target_product(self) -> int | None:
        if self.carrying is not None:
            return self.carrying
        for delivery in self.plan.deliveries:
            if self.product_status.get(delivery.product_id) == "pending":
                return delivery.product_id
        return None

    def begin(self) -> None:
        self.status = (
            f"Buscando P{self.plan.deliveries[0].product_id}"
            if self.plan.deliveries
            else "Sem rota"
        )

    def advance(self, simulated_seconds: float) -> None:
        if self.finished:
            return

        if self.route_index + 1 >= len(self.plan.route):
            self._finish(simulated_seconds)
            return

        self.route_index += 1
        self.steps += 1
        self.trail.append(self.position)

        event = self._events.get(self.route_index)
        if event:
            event_type, delivery = event
            if event_type == "pickup":
                self.carrying = delivery.product_id
                self.product_status[delivery.product_id] = "carrying"
                self.status = f"Levando P{delivery.product_id} para D{delivery.dock_id}"
            else:
                self.delivered_slots[delivery.dock_id] = delivery.product_id
                self.product_status[delivery.product_id] = "delivered"
                self.carrying = None
                next_product = self.target_product
                if next_product is None:
                    self.status = "Finalizando operação"
                else:
                    self.status = f"Buscando P{next_product}"

        if self.route_index + 1 >= len(self.plan.route):
            self._finish(simulated_seconds)

    def _finish(self, simulated_seconds: float) -> None:
        self.finished_at = simulated_seconds
        self.status = "Operação concluída"
