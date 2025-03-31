import numpy as np
import pygame
import time

# Parametry siatki
gridsize = 50  # Rozmiar siatki (NxN)
p_fire = 1   # Prawdopodobieństwo zapalenia sąsiedniego drzewa
p_tree = 0.7   # Prawdopodobieństwo początkowego istnienia drzewa
cell_size = 10 # Rozmiar komórki w pikselach

# Stany komórek
EMPTY = 0  
TREE = 1    
BURNING = 2  
BURNT = 3   

# Kolory
COLORS = {
    EMPTY: (200, 200, 200),  # Jasnoszary
    TREE: (34, 139, 34),     # Zieleń lasu
    BURNING: (255, 69, 0),   # Ogień (pomarańczowy)
    BURNT: (50, 50, 50)      # Spalone drzewo (ciemnoszary)
}

# Tworzenie początkowej siatki
forest = np.random.choice([EMPTY, TREE], size=(gridsize, gridsize), p=[1 - p_tree, p_tree])

# Zapalenie górnej krawędzi
forest[0, forest[0, :] == TREE] = BURNING

# Inicjalizacja Pygame
pygame.init()
screen = pygame.display.set_mode((gridsize * cell_size, gridsize * cell_size))
pygame.display.set_caption("Symulacja Rozprzestrzeniania Ognia")
clock = pygame.time.Clock()

def spread_fire(forest):
    """Rozprzestrzenia ogień w każdej iteracji."""
    new_forest = forest.copy()
    
    for x in range(gridsize - 1):  # Przetwarzamy od góry do dołu
        for y in range(gridsize):
            if forest[x, y] == BURNING:
                # Spalamy drzewo
                new_forest[x, y] = BURNT
                # Ogień rozprzestrzenia się na sąsiednie drzewa
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Von Neumann (4 sąsiadów)
                    if 0 <= x + dx < gridsize and 0 <= y + dy < gridsize:
                        if forest[x + dx, y + dy] == TREE and np.random.rand() < p_fire:
                            new_forest[x + dx, y + dy] = BURNING
    
    return new_forest

def draw_forest(forest):
    """Rysuje aktualny stan lasu."""
    screen.fill((0, 0, 0))  # Czarny background
    for x in range(gridsize):
        for y in range(gridsize):
            color = COLORS[forest[x, y]]
            pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))
    pygame.display.flip()

def run_simulation():
    """Uruchamia symulację w Pygame."""
    global forest
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        draw_forest(forest)
        pygame.time.delay(200)
        forest = spread_fire(forest)
        if not np.any(forest == BURNING):
            time.sleep(2)
            running = False
    
    pygame.quit()

# Uruchomienie symulacji
run_simulation()