import algoritmogenetico, aestrela
from pyamaze import maze

DIMENSAO = 100
INICIO = (DIMENSAO, DIMENSAO)
DESTINO = (1,1)

if __name__ == '__main__':
    labirinto = maze(DIMENSAO, DIMENSAO)
    labirinto.CreateMaze(loadMaze="maze.csv")
    aestrela.main(labirinto, DESTINO)

    labirinto = maze(DIMENSAO, DIMENSAO)
    labirinto.CreateMaze(loadMaze="maze.csv")
    algoritmogenetico.genetico(labirinto, INICIO, DESTINO)