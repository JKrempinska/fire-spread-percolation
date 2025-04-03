import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Definicja stanów komórek
EMPTY = 0      
TREE = 1       
BURNING = 2    
BURNED = 3     

# Kolory dla stanów
colors = ["white", "green", "red", "grey"]
cmap = ListedColormap(colors)

def initialize_forest(size, tree_density):
    """Inicjalizuje las z określoną gęstością drzew"""
    forest = np.random.choice([EMPTY, TREE], size=(size, size), p=[1-tree_density, tree_density])
    # Podpalamy drzewa w pierwszym rzędzie
    forest[0, :] = np.where(forest[0, :] == TREE, BURNING, EMPTY)
    return forest

def get_neighbors(x, y, size, neighborhood):
    """Zwraca sąsiadów komórki w zależności od typu sąsiedztwa"""
    if neighborhood == "von_neumann":
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    elif neighborhood == "moore":
        return [(x-1, y-1), (x-1, y), (x-1, y+1),
                (x, y-1), (x, y+1),
                (x+1, y-1), (x+1, y), (x+1, y+1)]
    return []

def spread_fire(forest, ignition_prob, neighborhood):
    """Symuluje rozprzestrzenianie się ognia na kolejną iterację"""
    new_forest = forest.copy()
    size = forest.shape[0]

    for x in range(size):
        for y in range(size):
            if forest[x, y] == BURNING:
                new_forest[x, y] = BURNED  # Spalone drzewo
                
                # Sprawdzamy sąsiadów
                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if 0 <= nx < size and 0 <= ny < size and forest[nx, ny] == TREE:
                        if np.random.rand() < ignition_prob:  # Szansa zapalenia
                            new_forest[nx, ny] = BURNING
    return new_forest

def animate_forest(size=50, tree_density=0.6, ignition_prob=0.3, neighborhood="von_neumann", save_gif=False):
    """Tworzy animację spalania lasu i opcjonalnie zapisuje jako GIF"""
    forest = initialize_forest(size, tree_density)
    
    fig, ax = plt.subplots()
    im = ax.imshow(forest, cmap=cmap, vmin=0, vmax=3)
    
    def update(frame):
        nonlocal forest
        forest = spread_fire(forest, ignition_prob, neighborhood)
        im.set_data(forest)
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=50, interval=200, repeat=False)
    
    if save_gif:
        ani.save("forest_fire.gif", writer=animation.PillowWriter(fps=10))
        print("GIF zapisany jako 'forest_fire.gif'")

    plt.show()

# Uruchomienie symulacji i zapisanie GIF-a
animate_forest(size=50, tree_density=0.6, ignition_prob=1, neighborhood="von_neumann", save_gif=False)

# Żeby zapisać większe siatki jako animacje trzeba zainstalować dodatkową bibliotekę 
# i nie jest to takie proste, więc chyba lepiej pokazać na żywo