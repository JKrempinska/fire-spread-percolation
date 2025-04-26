import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw

tree_cond = {"EMPTY": 0, "TREE": 1, "BURNING": 2, "BURNED": 3}
tree_color = {0: "white", 1: "green", 2: "red", 3: "grey"}

size = 50  # Rozmiar siatki
p_fire = 0.7   # Prawdopodobieństwo zapalenia sąsiedniego drzewa
p_tree = 0.5  # Prawdopodobieństwo początkowego istnienia drzewa / zalesienie

cmap = ListedColormap(tree_color.values())

def forest_grid(size, p_tree):
    forest = np.random.choice([tree_cond["EMPTY"], tree_cond["TREE"]], size=(size, size), p=[1-p_tree, p_tree])
    return forest

def get_neighbors(x, y, size, neighborhood):
    if neighborhood == "von_neumann":
        neighbors = np.array([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
    elif neighborhood == "moore":
        neighbors = np.array([(x-1, y-1), (x-1, y), (x-1, y+1),  
                            (x, y-1), (x, y+1),             
                            (x+1, y-1), (x+1, y), (x+1, y+1)])
    else:
        return [] 
    neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < size and 0 <= ny < size] 
    return neighbors

def start_fire(forest_before_fire):
    forest = forest_before_fire.copy()
    forest[0, :] = np.where(forest[0, :] == tree_cond["TREE"], tree_cond["BURNING"], forest[0, :])
    return forest

def spread_fire(forest_with_fire, burn_timer, p_fire, neighborhood, burn_time=4):
    size = forest_with_fire.shape[0]
    new_forest = forest_with_fire.copy()
    new_burn_timer = burn_timer.copy()

    for x in range(size):
        for y in range(size):
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_burn_timer[x, y] += 1

                if new_burn_timer[x, y] >= burn_time:
                    new_forest[x, y] = tree_cond["BURNED"]
                    new_burn_timer[x, y] = 0  # optional, just for clarity

                # Try to ignite neighbors
                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        if np.random.rand() < p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]

    return new_forest, new_burn_timer

def fire_simulation(size, p_fire, p_tree, neighborhood, gif_name, M_frames=100, burn_time=4):
    forest = forest_grid(size, p_tree)
    burn_timer = np.zeros_like(forest, dtype=int)
    frames = []


    for step in range(M_frames):
        if step == 0:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
            ax.imshow(forest, cmap=cmap, vmin=0, vmax=3)
            ax.axis('off')
            ax.set_title(f'Pusty las {size}x{size}, zalesienie = {p_tree}')
            plt.draw()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
            frames.append(image)
            plt.close(fig)

        elif step == 1:
            forest = start_fire(forest)
            fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
            ax.imshow(forest, cmap=cmap, vmin=0, vmax=3)
            ax.axis('off')
            ax.set_title(f'Siatka {size}x{size}, zalesienie = {p_tree}')
            plt.draw()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
            frames.append(image)
            plt.close(fig)

        else:
            forest, burn_timer = spread_fire(forest, burn_timer, p_fire, neighborhood, burn_time)
            fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
            ax.imshow(forest, cmap=cmap, vmin=0, vmax=3)
            ax.axis('off')
            ax.set_title(f'Siatka {size}x{size}, p_tree = {p_tree}, p_fire = {p_fire}, burn_time = {burn_time}, {neighborhood}')
            plt.draw()
            fig.canvas.draw()

            buf = fig.canvas.buffer_rgba()
            image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
            frames.append(image)
            plt.close(fig)

    frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
    print(f"Animacja została zapisana jako '{gif_name}.gif'")

def how_much_burned(size, p_fire, p_tree, neighborhood, M=100, burn_time=4):
    burned = np.zeros(M)
    forest = forest_grid(size, p_tree)
    total_trees = np.sum(forest == tree_cond["TREE"])
    forest = start_fire(forest)
    burn_timer = np.zeros_like(forest, dtype=int)
    
    for step in range(M):
        burned_trees = np.sum(forest == tree_cond["BURNED"]) + np.sum(forest == tree_cond["BURNING"])
        burned[step] = (burned_trees / total_trees) * 100 if total_trees > 0 else 0
        forest, burn_timer = spread_fire(forest, burn_timer, p_fire, neighborhood, burn_time)

    return burned, np.arange(M)


def critical_fire_threshold_burn_times(size, p_tree, neighborhood, M_steps=200, n_trials=10):
    p_fires = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0
    burn_times = [3, 4, 5]  # czasy spalania, które chcesz sprawdzić

    results = {burn_time: [] for burn_time in burn_times}

    for burn_time in burn_times:
        for p_fire in p_fires:
            trial_burned = []
            for _ in range(n_trials):
                burned, _ = how_much_burned(size, p_fire, p_tree, neighborhood, M_steps, burn_time)
                trial_burned.append(burned[-1])  # interesuje nas koniec symulacji
            avg_burn = np.mean(trial_burned)
            results[burn_time].append(avg_burn)

    # Rysowanie wykresów
    plt.figure(figsize=(10,7))
    for burn_time in burn_times:
        plt.plot(p_fires, results[burn_time], label=f'Czas spalania = {burn_time}')
    
    plt.xlabel('Prawdopodobieństwo zapalenia p_fire')
    plt.ylabel('Średni procent spalonych drzew (%)')
    plt.title(f'Próg krytyczny rozprzestrzeniania ognia ({neighborhood})')
    plt.grid(True)
    plt.legend()
    plt.show()

    return p_fires, results

# critical_fire_threshold_burn_times(size=100, p_tree=0.7, neighborhood="von_neumann")
# critical_fire_threshold_burn_times(size=100, p_tree=0.7, neighborhood="moore")

def fire_spread_speed_plot(size, p_tree, p_fires, neighborhoods, M_steps=100, n_trials=3, burn_time=1):
    plt.figure(figsize=(14, 6))

    for i, neighborhood in enumerate(neighborhoods):
        plt.subplot(1, len(neighborhoods), i+1)

        for p_fire in p_fires:
            all_burned = []
            for _ in range(n_trials):
                burned, steps = how_much_burned(size, p_fire, p_tree, neighborhood, M_steps, burn_time)
                all_burned.append(burned)
            avg_burned = np.mean(all_burned, axis=0)

            plt.plot(steps, avg_burned, label=f'p_fire={p_fire:.2f}')

        plt.title(f'Szybkość pożaru ({neighborhood})')
        plt.xlabel('Kroki symulacji')
        plt.ylabel('Procent spalonych drzew (%)')
        plt.ylim(0, 105)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

fire_spread_speed_plot(
    size=100,
    p_tree=0.7,
    p_fires=np.linspace(0.6, 1, 5) ,  # przykładowe wartości p_fire
    neighborhoods=["von_neumann", "moore"],
    M_steps=200,
    n_trials=10,
    burn_time=4
)

