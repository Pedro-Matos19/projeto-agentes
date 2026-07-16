import csv
import tracemalloc
from datetime import datetime
from pathlib import Path
from queue import PriorityQueue
from time import perf_counter

from pyamaze import agent, maze


destino = (1, 1)


def h_score(celula, destino):
    linhac, colunac = celula
    linhad, colunad = destino

    return abs(colunac - colunad) + abs(linhac - linhad)


def aestrela(labirinto, destino=(1, 1)):
    tracemalloc.start()
    inicio_tempo = perf_counter()

    f_score = {celula: float("inf") for celula in labirinto.grid}
    g_score = {}
    cel_inicial = (labirinto.rows, labirinto.cols)
    g_score[cel_inicial] = 0
    f_score[cel_inicial] = g_score[cel_inicial] + h_score(
        cel_inicial, destino
    )

    fila = PriorityQueue()
    item = (
        f_score[cel_inicial],
        h_score(cel_inicial, destino),
        cel_inicial,
    )
    fila.put(item)

    caminho = {}
    exploradas = set()
    nos_gerados = 1
    fronteira_maxima = 1
    encontrou_solucao = False

    while not fila.empty():
        celula = fila.get()[2]

        if celula in exploradas:
            continue

        exploradas.add(celula)

        if celula == destino:
            encontrou_solucao = True
            break

        for direcao in "NSWE":
            if labirinto.maze_map[celula][direcao] == 1:
                linha_celula, coluna_celula = celula

                if direcao == "N":
                    proxima_celula = (linha_celula - 1, coluna_celula)
                elif direcao == "S":
                    proxima_celula = (linha_celula + 1, coluna_celula)
                elif direcao == "W":
                    proxima_celula = (linha_celula, coluna_celula - 1)
                else:
                    proxima_celula = (linha_celula, coluna_celula + 1)

                novo_g_score = g_score[celula] + 1
                novo_f_score = novo_g_score + h_score(
                    proxima_celula, destino
                )

                if novo_f_score < f_score[proxima_celula]:
                    f_score[proxima_celula] = novo_f_score
                    g_score[proxima_celula] = novo_g_score
                    item = (
                        novo_f_score,
                        h_score(proxima_celula, destino),
                        proxima_celula,
                    )
                    fila.put(item)
                    caminho[proxima_celula] = celula
                    nos_gerados += 1

        fronteira_maxima = max(fronteira_maxima, fila.qsize())

    caminho_final = {}

    if encontrou_solucao:
        celula_analisada = destino

        while celula_analisada != cel_inicial:
            caminho_final[caminho[celula_analisada]] = celula_analisada
            celula_analisada = caminho[celula_analisada]

    tempo_execucao_ms = (perf_counter() - inicio_tempo) * 1000
    _, memoria_pico_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_celulas = len(labirinto.grid)
    estatisticas = {
        "algoritmo": "A*",
        "linhas": labirinto.rows,
        "colunas": labirinto.cols,
        "encontrou_solucao": encontrou_solucao,
        "passos": len(caminho_final) if encontrou_solucao else None,
        "custo": g_score.get(destino) if encontrou_solucao else None,
        "nos_expandidos": len(exploradas),
        "nos_gerados": nos_gerados,
        "fronteira_maxima": fronteira_maxima,
        "percentual_explorado": (len(exploradas) / total_celulas) * 100,
        "percentual_caminho": (len(caminho_final) / total_celulas) * 100,
        "tempo_execucao_ms": tempo_execucao_ms,
        "memoria_pico_kib": memoria_pico_bytes / 1024,
    }

    return caminho_final, estatisticas


def imprimir_estatisticas(estatisticas):
    print("\n========== ESTATÍSTICAS DO A* ==========")
    print(f"Labirinto: {estatisticas['linhas']} x {estatisticas['colunas']}")
    print(
        "Solução encontrada: "
        f"{'sim' if estatisticas['encontrou_solucao'] else 'não'}"
    )
    print(
        "Passos do caminho: "
        f"{estatisticas['passos'] if estatisticas['passos'] is not None else '-'}"
    )
    print(
        "Custo final: "
        f"{estatisticas['custo'] if estatisticas['custo'] is not None else '-'}"
    )
    print(f"Nós expandidos: {estatisticas['nos_expandidos']}")
    print(f"Nós gerados: {estatisticas['nos_gerados']}")
    print(f"Maior tamanho da fronteira: {estatisticas['fronteira_maxima']}")
    print(f"Percentual explorado: {estatisticas['percentual_explorado']:.2f}%")
    print(f"Percentual caminho final: {estatisticas['percentual_caminho']:.2f}%")
    print(f"Tempo da busca: {estatisticas['tempo_execucao_ms']:.3f} ms")
    print(f"Memória de pico: {estatisticas['memoria_pico_kib']:.2f} KiB")
    print("========================================\n")


def salvar_estatisticas(estatisticas, arquivo_saida="estatisticas.csv"):
    caminho_saida = Path(arquivo_saida)
    arquivo_existe = caminho_saida.exists()

    registro = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        **estatisticas,
    }

    with caminho_saida.open("a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=registro.keys())
        if not arquivo_existe:
            escritor.writeheader()
        escritor.writerow(registro)


def main():
    labirinto = maze(100, 100)
    labirinto.CreateMaze()

    agente = agent(labirinto, filled=True, footprints=True)
    caminho, estatisticas = aestrela(labirinto, destino)

    imprimir_estatisticas(estatisticas)
    salvar_estatisticas(estatisticas)

    if estatisticas["encontrou_solucao"]:
        labirinto.tracePath({agente: caminho}, delay=5)
    else:
        print("Não foi possível exibir um caminho porque não existe solução.")

    labirinto.run()


if __name__ == "__main__":
    main()
