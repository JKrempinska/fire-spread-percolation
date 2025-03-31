import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time

# Parametry siatki
gridsize = 50  # Rozmiar siatki (NxN)
p_fire = 0.6   # Prawdopodobieństwo zapalenia sąsiedniego drzewa

# Stany komórek
EMPTY = 0    # Puste pole 🌿
TREE = 1     # Drzewo 🌲
BURNING = 2  # Płonące drzewo 🔥
BURNT = 3    # Spalone drzewo 🖤

# Tworzenie początkowej siatki
forest = np.random.choice([EMPTY, TREE], size=(gridsize, gridsize), p=[0.3, 0.7])

# Zapalenie środkowego drzewa
mid = gridsize // 2
forest[mid, mid] = BURNING

# Kolory dla wizualizacji
cmap = mcolors.ListedColormap(["lightgray", "green", "red", "black"])
norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

def spread_fire(forest):
    """Rozprzestrzenia ogień w każdej iteracji."""
    new_forest = forest.copy()
    
    for x in range(1, gridsize - 1):
        for y in range(1, gridsize - 1):
            if forest[x, y] == BURNING:
                # Spalamy drzewo
                new_forest[x, y] = BURNT
                # Ogień rozprzestrzenia się na sąsiednie drzewa
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Von Neumann (4 sąsiadów)
                    if forest[x + dx, y + dy] == TREE and np.random.rand() < p_fire:
                        new_forest[x + dx, y + dy] = BURNING
    
    return new_forest

def run_simulation():
    """Uruchamia symulację i rysuje kolejne etapy."""
    global forest
    plt.figure(figsize=(6, 6))
    
    for _ in range(50):
        plt.clf()
        plt.imshow(forest, cmap=cmap, norm=norm)
        plt.axis("off")
        plt.pause(0.2)
        forest = spread_fire(forest)
        if not np.any(forest == BURNING):
            break
    plt.show()

# Uruchomienie symulacji
run_simulation()
