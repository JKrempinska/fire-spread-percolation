import numpy as np
import matplotlib.pyplot as plt

# Definicja stanów komórek
EMPTY = 0
TREE = 1
BURNING = 2
BURNED = 3

def initialize_forest(size, tree_density):
    """Tworzy siatkę lasu z określoną gęstością drzew"""
    forest = np.random.choice([EMPTY, TREE], size=(size, size), p=[1 - tree_density, tree_density])
    forest[0, :] = np.where(forest[0, :] == TREE, BURNING, EMPTY)
    return forest

def get_neighbors_von_neumann(x, y, size):
    """Zwraca sąsiadów komórki w sąsiedztwie von Neumanna"""
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def get_neighbors_moore(x, y, size): 
    """Zwraca sąsiadów komórki w sąsiedztwie Moorea"""
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1),
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]

def spread_fire(forest, neighbor_func):
    """Symuluje rozprzestrzenianie się ognia w jednej iteracji"""
    new_forest = forest.copy()
    size = forest.shape[0]
    for x in range(size):
        for y in range(size):
            if forest[x, y] == BURNING:
                new_forest[x, y] = BURNED
                for nx, ny in neighbor_func(x, y, size):
                    if 0 <= nx < size and 0 <= ny < size and forest[nx, ny] == TREE:
                        new_forest[nx, ny] = BURNING
    return new_forest

def fire_reaches_bottom(size, tree_density, neighbor_func, trials=50):
    """Sprawdza, dla jakiego prawdopodobieństwa drzewa ogień dociera do dołu siatki"""
    success_count = 0
    for _ in range(trials):
        forest = initialize_forest(size, tree_density)
        while BURNING in forest:
            forest = spread_fire(forest, neighbor_func)
        if np.any(forest[-1, :] == BURNED):
            success_count += 1
    return success_count / trials

# Parametry
sizes = [10, 20, 50, 100] # Rozmiary siatki
trials =100  # Liczba powtórzeń dla każdego tree_density
densities = np.arange(0.3, 0.85, 0.01) # Badamy różne tree_density od 0.3 do 0.85

# Progi teoretyczne
thresholds = [0.59275, 0.40725]  # von Neumann, Moore

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, neighbor_func, title, threshold in zip(
    axes,
    [get_neighbors_von_neumann, get_neighbors_moore],
    ["Sąsiedztwo von Neumanna (4)", "Sąsiedztwo Moore’a (8)"],
    thresholds
):
    for size in sizes:
        probs = [fire_reaches_bottom(size, d, neighbor_func, trials) for d in densities]
        ax.plot(densities, probs, label=f"{size}x{size}")
    ax.axhline(0.5, color='r', linestyle='--', label="Poziom 50%")
    ax.axvline(threshold, color='g', linestyle='--', label=f"Próg teoretyczny ≈ {threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Prawdopodobieństwo drzewa (tree_density)")
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel("Prawdopodobieństwo dotarcia ognia na dół")
plt.suptitle("Perkolacja ognia: von Neumann vs Moore (z progami teoretycznymi)")
plt.tight_layout()
plt.savefig("perkolacja_wykresy.png", dpi=300, bbox_inches='tight')
plt.show()
