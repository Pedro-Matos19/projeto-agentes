from __future__ import annotations

import random
import tracemalloc
from dataclasses import dataclass
from time import perf_counter

from aestrela import PathCache, build_route
from modelos import Cell, MetricValue, PlannerResult, ProgressCallback, Scenario


INF = 10**18


@dataclass(frozen=True)
class Chromosome:
    products: tuple[int, ...]
    slots: tuple[int, ...]


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int = 200
    generations: int = 300
    tournament_size: int = 3
    elite_rate: float = 0.05
    mutation_rate: float = 0.10
    stagnation_limit: int = 60


def ordered_crossover(
    first: tuple[int, ...],
    second: tuple[int, ...],
    rng: random.Random,
) -> tuple[int, ...]:

    size = len(first)
    if size < 2:
        return first
    left, right = sorted(rng.sample(range(size), 2))
    child: list[int | None] = [None] * size
    child[left : right + 1] = first[left : right + 1]
    remaining = iter(gene for gene in second if gene not in child)
    for index in list(range(right + 1, size)) + list(range(0, left)):
        child[index] = next(remaining)
    return tuple(int(gene) for gene in child)


def _swap_mutation(
    values: tuple[int, ...],
    mutation_rate: float,
    rng: random.Random,
) -> tuple[int, ...]:
    if len(values) < 2 or rng.random() >= mutation_rate:
        return values
    mutable = list(values)
    first, second = rng.sample(range(len(mutable)), 2)
    mutable[first], mutable[second] = mutable[second], mutable[first]
    return tuple(mutable)


def mutate_chromosome(
    chromosome: Chromosome,
    mutation_rate: float,
    rng: random.Random,
) -> Chromosome:
    return Chromosome(
        _swap_mutation(chromosome.products, mutation_rate, rng),
        _swap_mutation(chromosome.slots, mutation_rate, rng),
    )


class GeneticPlanner:
    algorithm = "Genético"

    def __init__(self, config: GeneticConfig | None = None) -> None:
        self.config = config or GeneticConfig()

    @staticmethod
    def _random_chromosome(
        product_ids: tuple[int, ...],
        slot_ids: tuple[int, ...],
        rng: random.Random,
    ) -> Chromosome:
        products = list(product_ids)
        slots = list(slot_ids)
        rng.shuffle(products)
        rng.shuffle(slots)
        return Chromosome(tuple(products), tuple(slots))

    @staticmethod
    def _distance(
        chromosome: Chromosome,
        start: Cell,
        products: dict[int, Cell],
        slots: dict[int, Cell],
        cache: PathCache,
    ) -> int | None:
        current = start
        total = 0
        for product_id, slot_id in zip(chromosome.products, chromosome.slots):
            to_product = cache.distance(current, products[product_id])
            to_slot = cache.distance(products[product_id], slots[slot_id])
            if to_product is None or to_slot is None:
                return None
            total += to_product + to_slot
            current = slots[slot_id]
        return total

    def _tournament(
        self,
        evaluated: list[tuple[int | None, Chromosome]],
        rng: random.Random,
    ) -> Chromosome:
        candidates = rng.sample(evaluated, min(self.config.tournament_size, len(evaluated)))
        return min(candidates, key=lambda item: item[0] if item[0] is not None else INF)[1]

    @staticmethod
    def _breed(first: Chromosome, second: Chromosome, rng: random.Random) -> Chromosome:
        return Chromosome(
            ordered_crossover(first.products, second.products, rng),
            ordered_crossover(first.slots, second.slots, rng),
        )

    def plan(
        self,
        scenario: Scenario,
        progress_callback: ProgressCallback | None = None,
    ) -> PlannerResult:
        tracemalloc.start()
        started_at = perf_counter()
        rng = random.Random(scenario.ga_seed)
        cache = PathCache(scenario.maze)
        product_ids = tuple(product.id for product in scenario.products)
        slot_ids = tuple(slot.id for slot in scenario.dock_slots)
        products = {product.id: product.position for product in scenario.products}
        slots = {slot.id: slot.position for slot in scenario.dock_slots}
        def distance(chromosome: Chromosome) -> int | None:
            return self._distance(chromosome, scenario.start, products, slots, cache)

        population = [
            self._random_chromosome(product_ids, slot_ids, rng)
            for _ in range(self.config.population_size)
        ]

        best_chromosome: Chromosome | None = None
        best_distance = INF
        best_generation = 0
        stagnation = 0
        generations_run = 0

        for generation in range(1, self.config.generations + 1):
            evaluated = [(distance(chromosome), chromosome) for chromosome in population]
            evaluated.sort(key=lambda item: item[0] if item[0] is not None else INF)
            generation_distance, generation_best = evaluated[0]
            generations_run = generation

            if generation_distance is not None and generation_distance < best_distance:
                best_distance = generation_distance
                best_chromosome = generation_best
                best_generation = generation
                stagnation = 0
            else:
                stagnation += 1

            if progress_callback:
                progress_callback(
                    {
                        "fase": "Genético",
                        "geracao": generation,
                        "geracoes_maximas": self.config.generations,
                        "melhor_distancia": 0 if best_distance == INF else best_distance,
                        "melhor_aptidao": 0.0 if best_distance == INF else 1 / (1 + best_distance),
                        "sem_melhoria": stagnation,
                    }
                )

            if stagnation >= self.config.stagnation_limit:
                break

            elite_count = max(1, round(self.config.population_size * self.config.elite_rate))
            next_population = [chromosome for _, chromosome in evaluated[:elite_count]]
            while len(next_population) < self.config.population_size:
                first = self._tournament(evaluated, rng)
                second = self._tournament(evaluated, rng)
                child = self._breed(first, second, rng)
                next_population.append(
                    mutate_chromosome(child, self.config.mutation_rate, rng)
                )
            population = next_population

        if best_chromosome is None:
            route = [scenario.start]
            deliveries = []
            success = False
            best_distance = 0
            product_order: tuple[int, ...] = ()
            slot_order: tuple[int, ...] = ()
        else:
            product_order = best_chromosome.products
            slot_order = best_chromosome.slots
            route, deliveries, success = build_route(
                scenario,
                product_order,
                slot_order,
                cache,
            )
            success = success and len(deliveries) == len(scenario.products)

        planning_ms = (perf_counter() - started_at) * 1000
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics: dict[str, MetricValue] = {
            **cache.metrics(),
            "passos": max(0, len(route) - 1),
            "geracoes": generations_run,
            "melhor_geracao": best_generation,
            "melhor_distancia": best_distance,
            "melhor_aptidao": 0.0 if not best_distance else 1 / (1 + best_distance),
            "sem_melhoria": stagnation,
            "ordem_produtos": "-".join(map(str, product_order)),
            "ordem_docas": "-".join(map(str, slot_order)),
            "memoria_pico_kib": peak_bytes / 1024,
        }
        return PlannerResult(
            algorithm=self.algorithm,
            route=route,
            deliveries=deliveries,
            planning_ms=planning_ms,
            metrics=metrics,
            success=success,
        )
