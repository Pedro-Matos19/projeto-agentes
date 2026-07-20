import random
from pyamaze import maze, agent, COLOR

TAMANHO_POPULACAO = 300       
COMPRIMENTO_CROMOSSOMO = 2500 
TAXA_MUTACAO = 0.03           
GERACOES = 1000
DIMENSAO = 100 
inicio = (DIMENSAO, DIMENSAO)
destino = (1, 1)               

DIRECOES = ['N', 'S', 'E', 'W']

def criar_individuo():
    return [random.choice(DIRECOES) for _ in range(COMPRIMENTO_CROMOSSOMO)]

def criar_populacao():
    return [criar_individuo() for _ in range(TAMANHO_POPULACAO)]

def calcular_fitness(individuo, m, start, goal):
    curr_x, curr_y = start
    visitados = {(curr_x, curr_y)}
    caminho_percorrido = [(curr_x, curr_y)]
    passos_uteis = 0

    for passo in individuo:
        if m.maze_map[(curr_x, curr_y)][passo] == 1:
            if passo == 'N': curr_x -= 1
            elif passo == 'S': curr_x += 1
            elif passo == 'E': curr_y += 1
            elif passo == 'W': curr_y -= 1
            
            if (curr_x, curr_y) not in visitados:
                visitados.add((curr_x, curr_y))
                passos_uteis += 1  
            
            caminho_percorrido.append((curr_x, curr_y))
        
        if (curr_x, curr_y) == goal:
            break

    distancia = abs(curr_x - goal[0]) + abs(curr_y - goal[1])
    
    pontuacao_distancia = 10000 / (distancia + 1)
    pontuacao_exploracao = passos_uteis * 1.5  

    fitness = pontuacao_distancia + pontuacao_exploracao
    return max(fitness, 1.0), caminho_percorrido

def selecao_torneio(populacao, fitness_scores, k=3):
    selecionados = random.sample(list(zip(populacao, fitness_scores)), k)
    return max(selecionados, key=lambda x: x[1])[0]

def crossover_dois_pontos(pai1, pai2):
    ponto1 = random.randint(1, COMPRIMENTO_CROMOSSOMO - 2)
    ponto2 = random.randint(ponto1 + 1, COMPRIMENTO_CROMOSSOMO - 1)
    
    filho1 = pai1[:ponto1] + pai2[ponto1:ponto2] + pai1[ponto2:]
    filho2 = pai2[:ponto1] + pai1[ponto1:ponto2] + pai2[ponto2:]
    return filho1, filho2

def mutar(individuo):
    for i in range(COMPRIMENTO_CROMOSSOMO):
        if random.random() < TAXA_MUTACAO:
            individuo[i] = random.choice(DIRECOES)
    return individuo

def labirinto(m, inicio, final, caminhos_historicos, melhor_caminho_geral ):
    dicionario_animacao = {}

    for caminho in caminhos_historicos:
        caminho_animado_hist = {}
        for idx in range(len(caminho) - 1):
            caminho_animado_hist[caminho[idx]] = caminho[idx + 1]
            
        agente_hist = agent(m, x=inicio[0], y=inicio[1], footprints=True, filled=True, color=COLOR.red)
        dicionario_animacao[agente_hist] = caminho_animado_hist

    caminho_animado_final = {}
    for idx in range(len(melhor_caminho_geral) - 1):
        caminho_animado_final[melhor_caminho_geral[idx]] = melhor_caminho_geral[idx + 1]

    agente_final = agent(m, x=inicio[0], y=inicio[1], footprints=True, filled=True)
    dicionario_animacao[agente_final] = caminho_animado_final

    m.tracePath(dicionario_animacao, delay=15) 
    m.run()

def genetico():
    m = maze(DIMENSAO, DIMENSAO)
    
    m.CreateMaze(loopPercent=100)
    
    populacao = criar_populacao()
    melhor_caminho_geral = []
    encontrou_solucao = False
    caminhos_historicos = []

    for geracao in range(GERACOES):
        avaliacoes = [calcular_fitness(ind, m, inicio, destino) for ind in populacao]
        fitness_scores = [score for score, _ in avaliacoes]
        caminhos = [path for _, path in avaliacoes]

        max_fitness = max(fitness_scores)
        melhor_indice = fitness_scores.index(max_fitness)
        melhor_individuo = populacao[melhor_indice]
        melhor_caminho_geral = caminhos[melhor_indice]
        
        posicao_final = melhor_caminho_geral[-1]
        distancia_restante = abs(posicao_final[0] - destino[0]) + abs(posicao_final[1] - destino[1])

        if (geracao + 1) % 5 == 0:
            caminhos_historicos.append(melhor_caminho_geral.copy())

        if (geracao + 1) % 10 == 0 or distancia_restante == 0:
            print(f"Geração {geracao+1:04d} | Melhor Fitness: {max_fitness:.1f} | Posição Final: {posicao_final} | Distância até Destino: {distancia_restante}")

        if posicao_final == destino:
            print(f"\nO Algoritmo Genético resolveu o labirinto de 100x100 na geração {geracao+1}!")
            encontrou_solucao = True
            break

        nova_populacao = []
        
        indices_ordenados = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
        for i in range(3):
            nova_populacao.append(populacao[indices_ordenados[i]])

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_torneio(populacao, fitness_scores)
            pai2 = selecao_torneio(populacao, fitness_scores)
            
            filho1, filho2 = crossover_dois_pontos(pai1, pai2)
            
            nova_populacao.append(mutar(filho1))
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(mutar(filho2))

        populacao = nova_populacao

    if not encontrou_solucao:
        print("\nO algoritmo encerrou as gerações. Exibindo a melhor rota parcial encontrada até o momento.")

    return m, caminhos_historicos, melhor_caminho_geral

if __name__ == '__main__':
    m, caminhos_historicos, melhor_caminho_geral = genetico()
    labirinto(m, inicio, destino, caminhos_historicos, melhor_caminho_geral)

    