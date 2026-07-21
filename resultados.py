from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from metricas import ComparisonMetrics, calculate_route_metrics, compare_plans
from modelos import AgentRuntimeState, Scenario


RUN_FIELDS = (
    "data_hora",
    "seed_cenario",
    "seed_genetico",
    "linhas",
    "colunas",
    "produtos",
    "algoritmo",
    "sucesso",
    "passos",
    "tempo_planejamento_ms",
    "tempo_simulado_s",
    "produtos_entregues",
    "curvas",
    "celulas_unicas",
    "revisitas",
    "percentual_revisita",
    "passos_por_entrega",
)
COMPARISON_FIELDS = (
    "gap_otimo_percentual",
    "sobreposicao_rotas_percentual",
    "concordancia_ordem_produtos_percentual",
    "concordancia_ordem_docas_percentual",
    "atingiu_otimo",
    "planejador_mais_rapido",
    "razao_tempo_planejamento",
)
ALGORITHM_FIELDS = (
    "nos_expandidos",
    "nos_gerados",
    "fronteira_maxima",
    "nos_expandidos_navegacao",
    "buscas_a_estrela_navegacao",
    "geracoes",
    "melhor_geracao",
    "melhor_aptidao",
    "sem_melhoria",
    "memoria_pico_kib",
    "ordem_produtos",
    "ordem_docas",
)
FIELDS = RUN_FIELDS + COMPARISON_FIELDS + ALGORITHM_FIELDS


def _comparison_values(
    comparison: ComparisonMetrics | None,
    is_genetic: bool,
) -> dict[str, object]:
    if comparison is None:
        return {field: "" for field in COMPARISON_FIELDS}
    return {
        "gap_otimo_percentual": (
            f"{comparison.optimality_gap_percent:.3f}" if is_genetic else "0.000"
        ),
        "sobreposicao_rotas_percentual": f"{comparison.route_overlap_percent:.3f}",
        "concordancia_ordem_produtos_percentual": (
            f"{comparison.product_order_agreement_percent:.3f}"
        ),
        "concordancia_ordem_docas_percentual": (
            f"{comparison.dock_order_agreement_percent:.3f}"
        ),
        "atingiu_otimo": comparison.ga_reached_optimum if is_genetic else True,
        "planejador_mais_rapido": comparison.faster_planner,
        "razao_tempo_planejamento": f"{comparison.planning_speed_ratio:.3f}",
    }


def save_results(
    scenario: Scenario,
    runtimes: dict[str, AgentRuntimeState],
    output: str | Path = "resultados_armazem.csv",
) -> None:
    path = Path(output)
    file_exists = _migrate_schema(path)
    timestamp = datetime.now().isoformat(timespec="seconds")
    comparison = (
        compare_plans(runtimes["astar"].plan, runtimes["genetic"].plan)
        if {"astar", "genetic"} <= runtimes.keys()
        else None
    )

    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        for runtime in runtimes.values():
            metrics = runtime.plan.metrics
            route_metrics = calculate_route_metrics(
                runtime.plan.route,
                len(runtime.plan.deliveries),
            )
            is_genetic = runtime.plan.algorithm == "Genético"
            record: dict[str, object] = {
                "data_hora": timestamp,
                "seed_cenario": scenario.seed,
                "seed_genetico": scenario.ga_seed,
                "linhas": scenario.rows,
                "colunas": scenario.cols,
                "produtos": len(scenario.products),
                "algoritmo": runtime.plan.algorithm,
                "sucesso": runtime.plan.success
                and runtime.delivered_count == len(scenario.products),
                "passos": runtime.steps,
                "tempo_planejamento_ms": f"{runtime.plan.planning_ms:.3f}",
                "tempo_simulado_s": f"{(runtime.finished_at or 0):.3f}",
                "produtos_entregues": runtime.delivered_count,
                "curvas": route_metrics.turns,
                "celulas_unicas": route_metrics.unique_cells,
                "revisitas": route_metrics.revisits,
                "percentual_revisita": f"{route_metrics.revisit_percent:.3f}",
                "passos_por_entrega": f"{route_metrics.steps_per_delivery:.3f}",
                **_comparison_values(comparison, is_genetic),
                **{field: metrics.get(field, "") for field in ALGORITHM_FIELDS},
            }
            writer.writerow(record)


def _migrate_schema(path: Path) -> bool:

    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        old_fields = tuple(reader.fieldnames or ())
        if old_fields == FIELDS:
            return True
        rows = list(reader)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return True
