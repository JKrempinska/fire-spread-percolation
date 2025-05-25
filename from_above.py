import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import label
from typing import List, Tuple
import inspect
import random
import seaborn as sns
import cv2

# stan drzew
tree_cond = {"EMPTY": 0, "TREE": 1, "BURNING": 2, "BURNED": 3, "WET_TREE": 4}

# kolory drzew
tree_color = {"EMPTY": "white", "TREE": "limegreen", "BURNING": "red", "BURNED": "grey", "WET_TREE": "blue"}

# rodzaje drzew
tree_type = {"EMPTY": 0, "Łatwopalne": 1, "Trudnopalne": 2}

# kolory rodzajów drzew
tree_type_color = {"EMPTY": "white", "Łatwopalne": "yellowgreen", "Trudnopalne": "mediumseagreen", }


size = 50  
p_fire = 0.7  
p_tree = 0.5  

cmap = ListedColormap(tree_color.values())
def forest_grid(size, p_tree):
    forest = np.random.choice([tree_cond["EMPTY"], tree_cond["TREE"]], size=(size, size), p=[1-p_tree, p_tree])
    return forest

grid = forest_grid(size, p_tree)

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

def spread_fire(forest_with_fire, size, p_fire, neighborhood):
    new_forest = forest_with_fire.copy()
    size = forest_with_fire.shape[0]

    for x in range(size):
        for y in range(size):
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_forest[x, y] = tree_cond["BURNED"]
                
                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        if np.random.rand() < p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]
    return new_forest

def start_fire_1st_row(forest_before_fire):
    forest = forest_before_fire.copy()
    forest[0, :] = np.where(forest[0, :] == tree_cond["TREE"], tree_cond["BURNING"], forest[0, :])
    return forest

def find_largest_cluster(forest, neighborhood):
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]]) if neighborhood == "von_neumann" else np.ones((3,3), int)
    tree_mask = (forest == tree_cond["TREE"])
    labeled, num_features = label(tree_mask, structure=structure)

    largest_size = 0
    largest_cluster = 0
    for i in range(1, num_features+1):
        size = np.sum(labeled == i)
        if size > largest_size:
            largest_size = size
            largest_cluster = i

    coords = np.argwhere(labeled == largest_cluster)
    return coords

def start_fire_claster(forest_before_fire, neighborhood="moore", mode="largest_single"):
    forest = forest_before_fire.copy()
    largest_cluster_coords = find_largest_cluster(forest, neighborhood)

    x, y = random.choice(largest_cluster_coords)
    forest[x, y] = tree_cond["BURNING"]

    return forest

# Rozpoczyna pożar w określonych współrzędnych (lista krotek).

def start_fire_coords(forest_before_fire, fire_coords=[(0, 0)]):
    forest = forest_before_fire.copy()
    for x, y in fire_coords:
        if forest[x, y] == tree_cond["TREE"]:
            forest[x, y] = tree_cond["BURNING"]
    return forest


p_fire_mixed = {"Łatwopalne": 1, "Trudnopalne": 0}
burning_time = {"Łatwopalne": 3, "Trudnopalne": 10}


def get_colored_forest(forest, mixed_forest):
    size = forest.shape[0]
    rgb_array = np.zeros((size, size, 3), dtype=np.uint8)

    inverse_tree_type = {v: k for k, v in tree_type.items()}

    for x in range(size):
        for y in range(size):
            cell = forest[x, y]

            if cell == tree_cond["TREE"]:
                t_type_value = mixed_forest[x, y]
                t_type_name = inverse_tree_type.get(t_type_value, "EMPTY")
                color_name = tree_type_color.get(t_type_name, "green")
            else:
                t_cond_name = [k for k, v in tree_cond.items() if v == cell][0]
                color_name = tree_color.get(t_cond_name, "white")

            rgb = np.array(mcolors.to_rgb(color_name)) * 255
            rgb_array[x, y] = rgb.astype(np.uint8)

    return rgb_array


def wczytaj_i_przetworz_zdjecie(sciezka, grid_size=10, tree_ratio={"Trudnopalne": 0.5, "Łatwopalne": 0.5}):
    tree_mapping = {
        "Trudnopalne": 1,
        "Łatwopalne": 2
    }

    img = cv2.imread(sciezka)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    wysokosc, szerokosc, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    maska_lasu = cv2.inRange(hsv, lower_green, upper_green)

    grid_rows = wysokosc // grid_size
    grid_cols = szerokosc // grid_size
    siatka = np.zeros((grid_rows, grid_cols), dtype=int)

    typy_drzew = list(tree_ratio.keys())
    prawdopodobienstwa = list(tree_ratio.values())

    for r in range(grid_rows):
        for c in range(grid_cols):
            roi = maska_lasu[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size]
            zielone_px = np.sum(roi > 0)
            if zielone_px > (grid_size**2 * 0.3):  # np. 30% zieleni to las
                typ = np.random.choice(typy_drzew, p=prawdopodobienstwa)
                siatka[r, c] = tree_mapping[typ]
            else:
                siatka[r, c] = 0  # brak lasu

    forest = np.where(siatka != tree_type["EMPTY"], tree_cond["TREE"], tree_cond["EMPTY"])

    return siatka, img, maska_lasu, forest

# def pokaz_wyniki(img, maska, siatka):
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.title("Zdjęcie oryginalne")
#     plt.imshow(img)
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title("Maska zieleni (obszary leśne)")
#     plt.imshow(maska, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title("Siatka lasu (1 = las)")
#     plt.imshow(siatka, cmap='Greens')
#     plt.colorbar()
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# sciezka_do_zdjecia = 'las3.jpg'  
# siatka, img, maska, forest = wczytaj_i_przetworz_zdjecie(sciezka_do_zdjecia, grid_size=2)

# pokaz_wyniki(img, maska, siatka)


def spread_fire_mixed(forest, mixed_forest, tree_types, p_fire_mixed,
                      neighborhood, burning_time, burning_time_dict,
                      rain_mask=None, rain_intensity=0, wet_timer=None, wet_time=3,
                      use_wind=False, wind_direction=None, wind_strength=0):

    forest, mixed_forest = forest, mixed_forest 
    burning_time = np.zeros_like(forest, dtype=int)
    frames = []
    size = forest.shape[0]
    new_forest = forest.copy()
    new_burning_time = burning_time.copy()
    new_wet_timer = wet_timer.copy() if wet_timer is not None else np.zeros_like(forest)

    id_to_name = {v: k for k, v in tree_types.items()}

    # Trzy wektory wpływane przez wiatr w zależności od kierunku
    wind_influence = {
        "S": [(-1, 0), (-1, -1), (-1, 1)], #switched
        "N": [(1, 0), (1, -1), (1, 1)],
        "W": [(0, 1), (-1, 1), (1, 1)],
        "E": [(0, -1), (-1, -1), (1, -1)],
        "SE": [(-1, -1), (-1, 0), (0, -1)],
        "SW": [(-1, 1), (-1, 0), (0, -1)],
        "NE": [(1, -1), (1, 0), (0, -1)],
        "NW": [(1, 1), (1, 0), (0, -1)]
    }
    main_wind = wind_influence.get(wind_direction.upper(), [])

    # Wektory przeciwwiatru
    reverse_wind = {
        "S": [(1, 0), (1, -1), (1, 1)],
        "N": [(-1, 0), (-1, -1), (-1, 1)],
        "W": [(0, -1), (-1, -1), (1, -1)],
        "E": [(0, 1), (-1, 1), (1, 1)],
        "SE": [(1, 1), (1, 0), (0, 1)],
        "SW": [(1, -1), (1, 0), (0, -1)],
        "NE": [(-1, 1), (-1, 0), (0, 1)],
        "NW": [(-1, -1), (-1, 0), (0, -1)]
    }
    opposite_vec = reverse_wind.get(wind_direction.upper(), (0, 0))

    for x in range(size):
        for y in range(size):
            if rain_mask is not None and rain_mask[x, y] == 0:
                if forest[x, y] == tree_cond["BURNING"] and np.random.rand() < rain_intensity:
                    new_forest[x, y] = tree_cond["BURNED"]
                    new_burning_time[x, y] = 0
                    continue
                if forest[x, y] == tree_cond["TREE"]:
                    new_wet_timer[x, y] = wet_time

            if forest[x, y] == tree_cond["BURNING"]:
                new_burning_time[x, y] += 1
                tree_name = id_to_name.get(mixed_forest[x, y], "EMPTY")

                if new_burning_time[x, y] >= burning_time_dict.get(tree_name, float('inf')):
                    new_forest[x, y] = tree_cond["BURNED"]

                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest[nx, ny] == tree_cond["TREE"]:
                        neighbor_name = id_to_name.get(mixed_forest[nx, ny], "EMPTY")
                        base_p_fire = p_fire_mixed.get(neighbor_name, 0)
                        reduction = rain_intensity if new_wet_timer[nx, ny] > 0 else 0
                        effective_p_fire = base_p_fire * (1 - reduction)

                        if use_wind and wind_direction is not None:
                            dx, dy = nx - x, ny - y
                            direction = (dx, dy)
                            if direction in main_wind:
                                effective_p_fire = min(1.0, effective_p_fire + wind_strength)
                            elif direction == opposite_vec:
                                effective_p_fire = max(0.0, effective_p_fire - wind_strength)

                        if np.random.rand() < effective_p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]
                            new_burning_time[nx, ny] = 0

            if new_wet_timer[x, y] > 0:
                new_wet_timer[x, y] -= 1

    return new_forest, new_burning_time, new_wet_timer

def get_colored_forest(forest, mixed_forest, tree_types, wet_timer=None):
    size = forest.shape[0]
    rgb_array = np.zeros((size, size, 3), dtype=np.uint8)
    inverse_tree_type = {v: k for k, v in tree_types.items()}

    for x in range(size):
        for y in range(size):
            cell = forest[x, y]
            if cell == tree_cond["TREE"]:
                t_type_value = mixed_forest[x, y]
                t_type_name = inverse_tree_type.get(t_type_value, "EMPTY")
                color_name = tree_type_color.get(t_type_name, "green")
                if wet_timer is not None and wet_timer[x, y] > 0:
                    color_name = tree_color["WET_TREE"]
            else:
                t_cond_name = [k for k, v in tree_cond.items() if v == cell][0]
                color_name = tree_color.get(t_cond_name, "white")

            rgb = np.array(mcolors.to_rgb(color_name)) * 255
            rgb_array[x, y] = rgb.astype(np.uint8)

    return rgb_array

def generate_rain_mask(siatka, rain_type="clouds", step=0, n_clouds=3, cloud_size=800, dx=-1, dy=2):
    size = min(np.shape(siatka))
    if rain_type == "random":
        rain_prob = 0.1
        mask = np.random.choice([0, 1], size=(size, size), p=[rain_prob, 1 - rain_prob])
        return mask

    mask = np.ones((size, size), dtype=int)
    np.random.seed(995)
    clouds = []

    for _ in range(n_clouds):
        start_x = np.random.randint(0, size)
        start_y = np.random.randint(0, size)
        cloud_cells = set()
        frontier = [(start_x, start_y)]

        while len(cloud_cells) < cloud_size and frontier:
            x, y = frontier.pop(np.random.randint(len(frontier)))
            if (x, y) in cloud_cells:
                continue
            cloud_cells.add((x, y))

            for nx, ny in get_neighbors(x, y, size, "moore"):
                if (nx, ny) not in cloud_cells and len(cloud_cells) < cloud_size:
                    if np.random.rand() < 0.5:
                        frontier.append((nx, ny))

        moved_cells = [((x + step * dx) % size, (y + step * dy) % size) for (x, y) in cloud_cells]
        clouds.extend(moved_cells)

    for x, y in clouds:
        mask[x, y] = 0

    return mask

def fire_simulation_mixed(mixed_forest, forest, tree_types, p_fire_mixed, neighborhood, burning_time_dict,
                          start_fire_func=start_fire_1st_row, fire_coords = [(120, 300)],
                          use_rain=False, rain_intensity=0, wet_time=None, rain_type='random', 
                          use_wind=False, wind_direction=None, wind_strength=0,
                          gif_name='fire', save=False, M_frames=100):
    
    forest, mixed_forest = forest, mixed_forest #mixed_forest_grid(size, p_tree, tree_types, tree_ratio=tree_ratio)
    size = forest.shape[0]

    burning_time = np.zeros_like(forest, dtype=int)
    wet_timer = np.zeros_like(forest, dtype=int)
    frames = []

    legend_elements = [mpatches.Patch(color=tree_type_color[label], label=label)
                       for label in tree_type if label != "EMPTY"]

    sig = inspect.signature(start_fire_func)
    required_params = sig.parameters.keys()
    start_fire_kwargs = {}

    if 'forest_before_fire' in required_params:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in required_params:
        start_fire_kwargs['neighborhood'] = neighborhood
    if 'fire_coords' in required_params:
        start_fire_kwargs['fire_coords'] = fire_coords

    for step in range(M_frames):
        rain_mask = np.ones((size, size), dtype=int)
        if use_rain and rain_intensity > 0:
            rain_mask = generate_rain_mask(mixed_forest, rain_type, step)

        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time, wet_timer = spread_fire_mixed(
                forest=forest,
                mixed_forest=mixed_forest,
                tree_types=tree_types,
                p_fire_mixed=p_fire_mixed,
                neighborhood=neighborhood,
                burning_time=burning_time,
                burning_time_dict=burning_time_dict,
                rain_mask=rain_mask,
                rain_intensity=rain_intensity,
                wet_timer=wet_timer,
                wet_time=wet_time,
                use_wind=use_wind,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        colored_forest = get_colored_forest(forest, mixed_forest, tree_types, wet_timer)
        ax.imshow(colored_forest)

        # Deszczowa nakładka
        if use_rain and rain_intensity > 0:
            rain_overlay = np.zeros((size, size, 4), dtype=np.uint8)
            rain_overlay[rain_mask == 0] = [0, 0, 255, 80]
            ax.imshow(rain_overlay)

        if use_wind and wind_strength > 0:
            wind_arrows = {
                "S": (0, 1),
                "N": (0, -1),
                "W": (1, 0),
                "E": (-1, 0),
                "NW": (-1, 1),
                "NE": (-1, -1),
                "SW": (1, 1),
                "SE": (-1, 1)
            }
            dx, dy = wind_arrows.get(wind_direction.upper(), (0, 0))

            if dx != 0 or dy != 0:
                ax.annotate('', xy=(0.95, 0.05), xytext=(0.95 - dx * 0.1, 0.05 - dy * 0.1),
                            xycoords='axes fraction', arrowprops=dict(facecolor='blue', edgecolor='black',
                                                                      linewidth=2, headwidth=10, headlength=10, shrink=0.01))
                ax.text(0.95, 0.02, 'Wiatr', transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=10, color='blue', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))

        ax.axis('off')
        title = f'Siatka {size}x{size}'
        if use_wind and wind_strength > 0:
            title += f' | \nWiatr: {wind_direction}, Siła wiatru: {wind_strength}'
        if use_rain and rain_intensity > 0:
            title += f' | \nIntensywność opadów: {rain_intensity}'

        ax.set_title(title, fontsize=14)
        ax.legend(handles=legend_elements, loc="lower left", title_fontsize=14, fontsize=12, borderpad=1.0,
                  bbox_to_anchor=(-0.1, -0.1), title="Rodzaje drzew")

        plt.draw()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)

    if save:
        frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
        print(f"Animacja została zapisana jako '{gif_name}.gif'")

# SIATKI ZE ZDJĘĆ
sciezka_do_zdjecia = 'las3.jpg'  
siatka, img, maska, forest = wczytaj_i_przetworz_zdjecie(sciezka_do_zdjecia, grid_size=2)

fire_simulation_mixed(mixed_forest=siatka, forest=forest, tree_types=tree_type, p_fire_mixed={"Łatwopalne": 0.7, "Trudnopalne": 0.3},
                      neighborhood="moore", burning_time_dict={"Łatwopalne": 2, "Trudnopalne": 4}, 
                      start_fire_func=start_fire_coords, fire_coords = [(120, 300)],
                      use_rain=True, rain_intensity=0.6, wet_time=3, rain_type="clouds",
                      use_wind=True, wind_direction="S", wind_strength=1,
                      gif_name="wind&rain_v2", save=True, M_frames=10)
