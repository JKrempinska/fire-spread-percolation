import numpy as np
import matplotlib.pyplot as plt

# Definicja stanów komórek
EMPTY = 0   
TREE = 1       
BURNING = 2   
BURNED = 3     

def initialize_forest(size, tree_density):
    """Tworzy siatkę lasu z określoną gęstością drzew"""
    forest = np.random.choice([EMPTY, TREE], size=(size, size), p=[1-tree_density, tree_density])
    forest[0, :] = np.where(forest[0, :] == TREE, BURNING, EMPTY)  # Podpalamy pierwszy rząd
    return forest

def get_neighbors(x, y, size):
    """Zwraca sąsiadów komórki w sąsiedztwie von Neumanna"""
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def spread_fire(forest):
    """Symuluje rozprzestrzenianie się ognia w jednej iteracji"""
    new_forest = forest.copy()
    size = forest.shape[0]

    for x in range(size):
        for y in range(size):
            if forest[x, y] == BURNING:
                new_forest[x, y] = BURNED  # Spalone drzewo
                for nx, ny in get_neighbors(x, y, size):
                    if 0 <= nx < size and 0 <= ny < size and forest[nx, ny] == TREE:
                        new_forest[nx, ny] = BURNING  # Sąsiad zawsze się zapala (prob = 1)
    return new_forest

def fire_reaches_bottom(size, tree_density, trials=50):
    """Sprawdza, dla jakiego prawdopodobieństwa drzewa ogień dociera do dołu siatki"""
    success_count = 0  # Liczba przypadków, gdzie ogień dotarł do dołu

    for _ in range(trials):
        forest = initialize_forest(size, tree_density)
        while BURNING in forest:
            forest = spread_fire(forest)
        if np.any(forest[-1, :] == BURNED):  # Czy w dolnym rzędzie są spalone drzewa?
            success_count += 1

    return success_count / trials  # Zwrot procentu przypadków, gdzie ogień dotarł na dół

# Parametry eksperymentu
size = 100  # Rozmiar siatki
trials = 50 # Liczba powtórzeń dla każdego tree_density
densities = np.arange(0.3, 1.0, 0.01)  # Badamy różne tree_density od 0.3 do 1.0
percolation_probs = [fire_reaches_bottom(size, d, trials) for d in densities]

# Wykres wyników
plt.figure(figsize=(8,6))
plt.plot(densities, percolation_probs, marker='o', linestyle='-', color='b', markersize=3)
plt.axhline(0.5, color='r', linestyle='--', label="Próg 50%")
plt.xlabel("Prawdopodobieństwo drzewa na siatce (tree_density)")
plt.ylabel("Prawdopodobieństwo dotarcia ognia na dół")
plt.title("Punkt perkolacji dla modelu rozprzestrzeniania się ognia")
plt.legend()
plt.grid(True)
plt.show()

# Znalezienie punktu perkolacji
percolation_threshold = densities[np.argmax(np.array(percolation_probs) > 0.5)]
print(f"🔹 Punkt perkolacji: tree_density ≈ {percolation_threshold:.2f}")
