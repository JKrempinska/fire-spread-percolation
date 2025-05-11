import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw

tree_cond = {"EMPTY": 0, "TREE": 1, "BURNING": 2, "BURNED": 3}

size = 50  # Rozmiar siatki
p_fire = 0.7   # Prawdopodobieństwo zapalenia sąsiedniego drzewa
p_tree = 0.5  # Prawdopodobieństwo początkowego istnienia drzewa / zalesienie

tree_color = {
    0: "white",        # EMPTY
    1: "limegreen",        # TREE
    2: "red",          # BURNING
    3: "grey",         # BURNED
    4: "forestgreen"       # WET_TREE (turkus)
}

cmap = ListedColormap(tree_color.values())


def forest_grid(size, p_tree):
    forest = np.random.choice([tree_cond["EMPTY"], tree_cond["TREE"]], size=(size, size), p=[1-p_tree, p_tree])
    return forest


def generate_rain_mask(size, rain_type="clouds", step=0, n_clouds = 3, cloud_size = 800, dx=-1, dy=2):

    if rain_type == "random":
        # Losowe komórki z deszczem
        rain_prob = 0.1  # 10% komórek losowo z deszczem
        mask = np.random.choice([0, 1], size=(size, size), p=[rain_prob, 1 - rain_prob])
        return mask

    mask = np.ones((size, size), dtype=int)

    np.random.seed(995)  # Stała losowość chmur

    clouds = []

    for _ in range(n_clouds):
        # Losowy punkt początkowy chmury
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)

        # BFS/random walk tworzący nieregularną chmurę
        cloud_cells = set()
        frontier = [(start_x, start_y)]

        while len(cloud_cells) < cloud_size and frontier:
            x, y = frontier.pop(np.random.randint(len(frontier)))
            if (x, y) in cloud_cells:
                continue
            cloud_cells.add((x, y))

            for nx, ny in get_neighbors(x, y, size, "moore"):
                if (nx, ny) not in cloud_cells and len(cloud_cells) < cloud_size:
                    if np.random.rand() < 0.5:  # 50% szans na "doklejenie"
                        frontier.append((nx, ny))

        # Przesunięcie chmury o krok * (dx, dy)
        moved_cells = [((x + step * dx) % size, (y + step * dy) % size) for (x, y) in cloud_cells]
        clouds.extend(moved_cells)

    for x, y in clouds:
        mask[x, y] = 0  # Pada deszcz

    return mask



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

def spread_fire(forest_with_fire, burn_timer, p_fire, neighborhood, burn_time=1,
                rain_intensity=0, rain_mask=None, wet_timer=None, wet_time=3):  # Dodajemy nowy argument dla prawdopodobieństwa gaszenia
    size = forest_with_fire.shape[0]
    new_forest = forest_with_fire.copy()
    new_burn_timer = burn_timer.copy()
    new_wet_timer = wet_timer.copy()

    for x in range(size):
        for y in range(size):
            # GAŚ ogień z pewnym prawdopodobieństwem, jeśli pada deszcz
            if rain_mask is not None and rain_mask[x, y] == 0 and forest_with_fire[x, y] == tree_cond["BURNING"]:
                # Prawdopodobieństwo zgaszenia ognia zależne od intensywności deszczu
                if np.random.rand() < rain_intensity:
                    new_forest[x, y] = tree_cond["BURNED"]
                    new_burn_timer[x, y] = 0
                    continue

            # OZNACZ drzewa jako mokre
            if rain_mask is not None and rain_mask[x, y] == 0 and forest_with_fire[x, y] == tree_cond["TREE"]:
                new_wet_timer[x, y] = wet_time

            # AKTUALIZUJ wilgotność
            if new_wet_timer[x, y] > 0:
                new_wet_timer[x, y] -= 1

            # PRZETWARZANIE stanu BURNING
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_burn_timer[x, y] += 1

                if new_burn_timer[x, y] >= burn_time:
                    new_forest[x, y] = tree_cond["BURNED"]
                    new_burn_timer[x, y] = 0

                # Próbuj podpalić sąsiadów
                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        # Jeśli drzewo jest mokre — zmniejsz p_fire
                        reduction = rain_intensity if new_wet_timer[nx, ny] > 0 else 0
                        effective_p_fire = p_fire * (1 - reduction)
                        if np.random.rand() < effective_p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]

    return new_forest, new_burn_timer, new_wet_timer

def fire_simulation(size, p_fire, p_tree, neighborhood, gif_name,
                    M_frames=100, burn_time=1, rain_intensity=0.5,
                    rain_type="static", wet_time=3):
    forest = forest_grid(size, p_tree)
    burn_timer = np.zeros_like(forest, dtype=int)
    wet_timer = np.zeros_like(forest, dtype=int)
    frames = []

    for step in range(M_frames):
        rain_mask = generate_rain_mask(size, rain_type, step)

        if step == 0:
            forest = start_fire(forest)
        else:
            forest, burn_timer, wet_timer = spread_fire(
                forest, burn_timer, p_fire, neighborhood, burn_time,
                rain_intensity, rain_mask, wet_timer, wet_time
            )

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        display_forest = forest.copy()
        display_forest[(forest == tree_cond["TREE"]) & (wet_timer > 0)] = 4
        ax.imshow(display_forest, cmap=cmap, vmin=0, vmax=4)

        # Nakładka z deszczem
        rain_overlay = np.zeros((size, size, 4), dtype=np.uint8)
        rain_overlay[rain_mask == 0] = [0, 0, 255, 80]  # Niebieski deszcz
        ax.imshow(rain_overlay)

        ax.axis('off')
        ax.set_title(f'Step {step}')
        plt.draw()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)

    frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
    print(f"Animacja została zapisana jako '{gif_name}.gif'")




fire_simulation(size=100, p_fire=0.6, p_tree=0.6, neighborhood="moore",
                gif_name="rain_effects_2", M_frames=250, burn_time=3,
                rain_intensity=0.8, rain_type="random", wet_time=10)




