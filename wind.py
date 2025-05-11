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

tree_cond = {"EMPTY": 0, "TREE": 1, "BURNING": 2, "BURNED": 3}
tree_color = {"EMPTY": "white", "TREE": "green", "BURNING": "red", "BURNED": "grey"}
#### 2. Poczatkowe parametry siatki 2D.
size = 50  
p_fire = 0.7  
p_tree = 0.5  
#### 3. Początkowa siatka 2D (jeszcze przed pożarem).
cmap = ListedColormap(tree_color.values())
def forest_grid(size, p_tree):
    forest = np.random.choice([tree_cond["EMPTY"], tree_cond["TREE"]], size=(size, size), p=[1-p_tree, p_tree])
    return forest
grid = forest_grid(size, p_tree)


#### 4. Sąsiedztwo (Von Neumanna i Moore'a)
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

#### 5. Pożar i jego rozprzestrzenianie się

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
#### 6. Symulacja rozprzestrzeniania się ognia
def fire_simulation(size, p_fire, p_tree, neighborhood, gif_name, M_frames=100):
    forest = forest_grid(size, p_tree)
    frames = []

    for step in range(M_frames):
        if step == 1:
            forest = start_fire(forest)
        elif step > 1:
            forest = spread_fire(forest, forest, p_fire, neighborhood)

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        ax.imshow(forest, cmap=cmap, vmin=0, vmax=3)
        ax.axis('off')
        ax.set_title(f'Siatka {size}x{size}, zalesienie = {p_tree}, sąsiedztwo = {neighborhood}')
        plt.draw()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)

    frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
    print(f"Animacja została zapisana jako '{gif_name}.gif'")

def how_much_burned(size, p_fire, p_tree, neighborhood, M=100):
    burned = np.zeros(M)
    forest = forest_grid(size, p_tree)
    total_trees = np.sum(forest == tree_cond["TREE"])
    forest = start_fire(forest)
    
    for step in range(M):
        burned_trees = np.sum(forest == tree_cond["BURNED"]) + np.sum(forest == tree_cond["BURNING"])
        burned[step] = (burned_trees / total_trees) * 100 if total_trees > 0 else 0
        forest = spread_fire(forest, size, p_fire, neighborhood)

    return burned, np.arange(M)
##### 7.1. Rozprzestrzenianie się pożaru VS n-ty krok symulacji

M = 200
p_fire, p_tree = 1, 0.6
size = 50


N = 100
M = 200
p_fire, p_tree = 1, 0.6
size = 50


M = 200
p_fire = 1
p_tree_values = np.arange(0.1, 1.1, 0.1)
size = 50

N = 100
M = 200
p_fire = 1
p_tree_values = np.arange(0.1, 1.1, 0.1)
size = 50

M = 200
p_fire, p_tree = 1, 0.6
size_values = np.arange(50, 300, 50)

M = 200
N = 100
p_fire, p_tree = 1, 0.6
size_values = np.arange(50, 300, 50)


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


# #### 2. Las nie jest jednolity - mogą występować różne rodzaje drzew co wpływa na łatwość ich zapalenia (p_fire). Na ten moment rozważamy najprosztszą sytuację:

tree_type = {"EMPTY": 0, "Łatwopalne": 1, "Trudnopalne": 2}
tree_type_color = {"EMPTY": "white", "Łatwopalne": "yellowgreen", "Trudnopalne": "green"}
p_fire_mixed = {"Łatwopalne": 1, "Trudnopalne": 0}
burning_time = {"Łatwopalne": 3, "Trudnopalne": 10}

def mixed_forest_grid(size: int, p_tree: float, tree_types: List[int] = None, tree_ratio: List[float] = [0.3, 0.7]) -> Tuple[np.ndarray, np.ndarray]:
    if tree_types is None:
        tree_types = list(tree_type.values())
    #tree_ratio = {k: v * p_tree for k, v in tree_ratio.items()}

    tree_ratio = np.array(tree_ratio ) * p_tree

    mixed_forest = np.random.choice(tree_types, size=(size,size), p=[1 - p_tree, *tree_ratio])
    forest = np.where(mixed_forest != tree_type["EMPTY"], tree_cond["TREE"], tree_cond["EMPTY"])

    return forest, mixed_forest
##### 2.1. Siatka lasu z różnymi typami drzew
def draw_mixed_forest(size, tree_ratio, colors, tree_types=None, p_tree=0.6):
    if tree_types is None:
        tree_types = list(tree_type.values())
    legend_labels=list(tree_type.keys())[1:]

    _, mixed_grid = mixed_forest_grid(size=size, p_tree=p_tree, tree_types=tree_types, tree_ratio=tree_ratio)
    cmap = ListedColormap(colors.values())
    vmin, vmax = 0, len(list(colors.keys()))-1

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ax.imshow(mixed_grid, cmap=cmap, vmin=vmin, vmax=vmax)

    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.title(f'zalesienie = {p_tree}, tree_ratio = {tree_ratio}')

    legend_elements = [mpatches.Patch(color=colors[label], label=label) for label in legend_labels]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.6, 1.0), title="Rodzaje drzew: ")
    
    plt.show()
size = 50
p_tree = 0.8
ratio =[0.3, 0.7]

# draw_mixed_forest(size=size, tree_ratio=ratio, colors=tree_type_color, p_tree=p_tree)
##### 2.2. Rozprzestrzenianie się pożaru w lesie z różnymi typami drzew
def spread_fire_mixed(forest_with_fire, mixed_forest, p_fire_mixed, neighborhood, burning_time, burning_time_dict):
    size = forest_with_fire.shape[0]
    new_forest = forest_with_fire.copy()
    new_burning_time = burning_time.copy()

    id_to_name = {v: k for k, v in tree_type.items()}

    for x in range(size):
        for y in range(size):
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_burning_time[x, y] += 1
                tree_name = id_to_name.get(mixed_forest[x, y], "EMPTY")

                if new_burning_time[x, y] >= burning_time_dict.get(tree_name, float('inf')):
                    new_forest[x, y] = tree_cond["BURNED"]

                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        neighbor_name = id_to_name.get(mixed_forest[nx, ny], "EMPTY")
                        p_fire = p_fire_mixed.get(neighbor_name, 0)
                        if np.random.rand() < p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]
                            new_burning_time[nx, ny] = 0

    return new_forest, new_burning_time

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

def fire_simulation_mixed(size, p_tree, tree_ratio, p_fire_mixed, neighborhood, burning_time_dict, 
                          start_fire_func=start_fire_1st_row, gif_name='fire', save=False, M_frames=100):
    
    forest, mixed_forest = mixed_forest_grid(size, p_tree, tree_ratio=tree_ratio)
    burning_time = np.zeros_like(forest, dtype=int)
    frames = []

    legend_elements = [mpatches.Patch(color=tree_type_color[label], label=label)
                       for label in tree_type][1:]

    sig = inspect.signature(start_fire_func)
    required_params = sig.parameters.keys()
    
    start_fire_kwargs = {}
    if 'forest_before_fire' in required_params:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in required_params:
        start_fire_kwargs['neighborhood'] = neighborhood

    for step in range(M_frames):
        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time = spread_fire_mixed(forest, mixed_forest, p_fire_mixed, neighborhood, burning_time, burning_time_dict)

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        colored_forest = get_colored_forest(forest, mixed_forest)
        ax.imshow(colored_forest)
        ax.axis('off')
        ax.set_title(f'Siatka {size}x{size}, zalesienie = {p_tree}, sąsiedztwo = {neighborhood}')
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

#DODANIE WIATRU

def spread_fire_mixed_wind(forest_with_fire, mixed_forest, p_fire_mixed, neighborhood, 
                            burning_time, burning_time_dict, wind_direction="N", wind_strength=0.2):
    size = forest_with_fire.shape[0]
    new_forest = forest_with_fire.copy()
    new_burning_time = burning_time.copy()

    id_to_name = {v: k for k, v in tree_type.items()}

    # Wektory kierunków
    direction_vectors = {
        "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
        "NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)
    }
    wind_vec = direction_vectors.get(wind_direction, (0, 0))
    opposite_vec = (-wind_vec[0], -wind_vec[1])

    for x in range(size):
        for y in range(size):
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_burning_time[x, y] += 1
                tree_name = id_to_name.get(mixed_forest[x, y], "EMPTY")

                if new_burning_time[x, y] >= burning_time_dict.get(tree_name, float('inf')):
                    new_forest[x, y] = tree_cond["BURNED"]

                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        dx, dy = nx - x, ny - y
                        direction = (dx, dy)

                        neighbor_name = id_to_name.get(mixed_forest[nx, ny], "EMPTY")
                        base_p_fire = p_fire_mixed.get(neighbor_name, 0)

                        # Modyfikacja prawdopodobieństwa zapłonu w zależności od kierunku wiatru
                        if direction == wind_vec:
                            p_fire = min(1.0, base_p_fire + wind_strength)
                        elif direction == opposite_vec:
                            p_fire = max(0.0, base_p_fire - wind_strength)
                        else:
                            p_fire = base_p_fire

                        if np.random.rand() < p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]
                            new_burning_time[nx, ny] = 0

    return new_forest, new_burning_time


def fire_simulation_mixed_wind(size, p_tree, tree_ratio, p_fire_mixed, neighborhood, burning_time_dict,
                                wind_direction="N", wind_strength=0.2,
                                start_fire_func=start_fire_claster, gif_name='fire_wind',
                                save=False, M_frames=100):

    forest, mixed_forest = mixed_forest_grid(size, p_tree, tree_ratio=tree_ratio)
    burning_time = np.zeros_like(forest, dtype=int)
    frames = []

    legend_elements = [mpatches.Patch(color=tree_type_color[label], label=label)
                       for label in tree_type][1:]

    sig = inspect.signature(start_fire_func)
    required_params = sig.parameters.keys()
    
    start_fire_kwargs = {}
    if 'forest_before_fire' in required_params:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in required_params:
        start_fire_kwargs['neighborhood'] = neighborhood

    for step in range(M_frames):
        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time = spread_fire_mixed_wind(
                forest_with_fire=forest,
                mixed_forest=mixed_forest,
                p_fire_mixed=p_fire_mixed,
                neighborhood=neighborhood,
                burning_time=burning_time,
                burning_time_dict=burning_time_dict,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        colored_forest = get_colored_forest(forest, mixed_forest)
        ax.imshow(colored_forest)
        ax.axis('off')

        # Tytuł
        ax.set_title(f'Siatka {size}x{size}, Wiatr: {wind_direction}, Siła: {wind_strength}')

        # Legenda
        ax.legend(handles=legend_elements, loc="lower left", title_fontsize=14, fontsize=12, borderpad=1.0,
                  bbox_to_anchor=(-0.1, -0.1), title="Rodzaje drzew")

        # Strzałka wiatru
        arrow_length = 10

                # Rysowanie strzałki wiatru – w prawym górnym rogu (transAxes => współrzędne od 0 do 1)
        wind_vectors = {
            "N": (0, -1),
            "NE": (0.7, -0.7),
            "E": (1, 0),
            "SE": (0.7, 0.7),
            "S": (0, 1),
            "SW": (-0.7, 0.7),
            "W": (-1, 0),
            "NW": (-0.7, -0.7),
        }
        dx, dy = wind_vectors.get(wind_direction.upper(), (0, 0))

        if dx != 0 or dy != 0:
            ax.annotate(
                '', 
                xy=(0.95, 0.05), 
                xytext=(0.95 - dx*0.1, 0.05 - dy*0.1),
                xycoords='axes fraction',
                arrowprops=dict(
                    facecolor='blue', edgecolor='black',
                    linewidth=2, headwidth=12, headlength=10, shrink=0.01
                )
            )
            # Etykieta "Wiatr"
            ax.text(
                0.95, 0.02, 'Wiatr',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10, color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue')
            )

        plt.draw()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)
        
    if save:
        frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
        print(f"Animacja została zapisana jako '{gif_name}.gif'")



def spread_fire_mixed_wind(forest_with_fire, mixed_forest, p_fire_mixed, neighborhood, 
                            burning_time, burning_time_dict, wind_direction="N", wind_strength=0.2):
    size = forest_with_fire.shape[0]
    new_forest = forest_with_fire.copy()
    new_burning_time = burning_time.copy()

    id_to_name = {v: k for k, v in tree_type.items()}

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

    # Wektory przeciwwiatru (główne przeciwne)
    reverse_wind = {
        "S": (1, 0),
        "N": (-1, 0),
        "W": (0, -1),
        "E": (0, 1),
        "NW": (-1, -1),
        "NE": (-1, 1),
        "SW": (1, -1),
        "SE": (1, 1)
    }
    opposite_vec = reverse_wind.get(wind_direction.upper(), (0, 0))

    for x in range(size):
        for y in range(size):
            if forest_with_fire[x, y] == tree_cond["BURNING"]:
                new_burning_time[x, y] += 1
                tree_name = id_to_name.get(mixed_forest[x, y], "EMPTY")

                if new_burning_time[x, y] >= burning_time_dict.get(tree_name, float('inf')):
                    new_forest[x, y] = tree_cond["BURNED"]

                for nx, ny in get_neighbors(x, y, size, neighborhood):
                    if forest_with_fire[nx, ny] == tree_cond["TREE"]:
                        dx, dy = nx - x, ny - y
                        direction = (dx, dy)

                        neighbor_name = id_to_name.get(mixed_forest[nx, ny], "EMPTY")
                        base_p_fire = p_fire_mixed.get(neighbor_name, 0)

                        # Modyfikacja p_fire
                        if direction in main_wind:
                            p_fire = min(1.0, base_p_fire + wind_strength)
                        elif direction == opposite_vec:
                            p_fire = max(0.0, base_p_fire - wind_strength)
                        else:
                            p_fire = base_p_fire

                        if np.random.rand() < p_fire:
                            new_forest[nx, ny] = tree_cond["BURNING"]
                            new_burning_time[nx, ny] = 0

    return new_forest, new_burning_time

def fire_simulation_mixed_wind(size, p_tree, tree_ratio, p_fire_mixed, neighborhood, burning_time_dict,
                                wind_direction="N", wind_strength=0.2,
                                start_fire_func=start_fire_claster,
                                save=False, M_frames=100, gif_name='sim_wind'):

    forest, mixed_forest = mixed_forest_grid(size, p_tree, tree_ratio=tree_ratio)
    burning_time = np.zeros_like(forest, dtype=int)
    frames = []

    legend_elements = [mpatches.Patch(color=tree_type_color[label], label=label)
                       for label in tree_type][1:]

    sig = inspect.signature(start_fire_func)
    required_params = sig.parameters.keys()

    start_fire_kwargs = {}
    if 'forest_before_fire' in required_params:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in required_params:
        start_fire_kwargs['neighborhood'] = neighborhood

    for step in range(M_frames):
        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time = spread_fire_mixed_wind(
                forest_with_fire=forest,
                mixed_forest=mixed_forest,
                p_fire_mixed=p_fire_mixed,
                neighborhood=neighborhood,
                burning_time=burning_time,
                burning_time_dict=burning_time_dict,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        colored_forest = get_colored_forest(forest, mixed_forest)
        ax.imshow(colored_forest)
        ax.axis('off')

        # Tytuł
        ax.set_title(f'Siatka {size}x{size}, Wiatr: {wind_direction}, Siła: {wind_strength}')

        # Legenda
        ax.legend(handles=legend_elements, loc="lower left", title_fontsize=14, fontsize=12, borderpad=1.0,
                  bbox_to_anchor=(-0.1, -0.1), title="Rodzaje drzew")

        # Strzałki tylko dla głównych kierunków
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
            ax.annotate(
                '', 
                xy=(0.95, 0.05), 
                xytext=(0.95 - dx * 0.1, 0.05 - dy * 0.1),
                xycoords='axes fraction',
                arrowprops=dict(
                    facecolor='blue', edgecolor='black',
                    linewidth=2, headwidth=10, headlength=10, shrink=0.01
                )
            )
            ax.text(
                0.95, 0.02, 'Wiatr',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10, color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue')
            )

        plt.draw()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)

    if save:
        frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200) #f'{gif_name}.gif'
        print(f"Animacja została zapisana jako '{gif_name}.gif'")


fire_simulation_mixed_wind(
    size=100,
    p_tree=0.7,
    tree_ratio=[1, 0],
    p_fire_mixed={"Łatwopalne": 0.6, "Trudnopalne": 0.3},
    neighborhood="moore",
    burning_time_dict={"Łatwopalne": 2, "Trudnopalne": 1},
    wind_direction="SW",  # Kierunek wiatru
    wind_strength=0.5,   # Siła wiatru
    M_frames=150,
    save=True,
    gif_name=f'C:/Users/Ala/Desktop/wiatr/sim_wind_sw'
)

fire_simulation_mixed_wind(
    size=100,
    p_tree=0.6,
    tree_ratio=[0.7, 0.3],
    p_fire_mixed={"Łatwopalne": 0.6, "Trudnopalne": 0.3},
    neighborhood="moore",
    burning_time_dict={"Łatwopalne": 2, "Trudnopalne": 1},
    wind_direction="W",  # Kierunek wiatru
    wind_strength=0.3,   # Siła wiatru
    M_frames=150,
    save=True,
    gif_name=f'C:/Users/Ala/Desktop/wiatr/sim_wind_w'
)

## ZMIANA KIERUNKU W TREAKCIE

def mixed_forest_grid(size: int,
                       p_tree: float,
                       tree_ratio: dict[str, float] = None,
                       tree_types: dict[str, int] = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    # Domyślne typy drzew
    if tree_types is None:
        tree_types = {k: v for k, v in tree_type.items() if k != "EMPTY"}

    # Domyślne proporcje: równe udziały
    if tree_ratio is None:
        n = len(tree_types)
        tree_ratio = {name: 1/n for name in tree_types}

    # Skalowanie proporcji przez p_tree
    scaled = {name: ratio * p_tree for name, ratio in tree_ratio.items()}
    empty_prob = 1 - sum(scaled.values())

    # Budujemy listę kodów i prawdopodobieństw
    labels = [tree_type["EMPTY"]] + [tree_type[name] for name in tree_types]
    probs = [empty_prob] + [scaled[name] for name in tree_types]

    mixed_forest = np.random.choice(labels, size=(size, size), p=probs)
    forest = np.where(mixed_forest != tree_type["EMPTY"], tree_cond["TREE"], tree_cond["EMPTY"])

    return forest, mixed_forest


def fire_simulation_mixed_wind(size, p_tree, tree_ratio, p_fire_mixed, neighborhood, burning_time_dict,
                                wind_direction="N", wind_strength=0.2,
                                start_fire_func=start_fire_claster,
                                save=False, M_frames=100, gif_name='sim_wind',
                                wind_change_interval=None):

    forest, mixed_forest = mixed_forest_grid(size, p_tree, tree_ratio=tree_ratio)
    burning_time = np.zeros_like(forest, dtype=int)
    frames = []

    legend_elements = [mpatches.Patch(color=tree_type_color[label], label=label)
                       for label in tree_type][1:]

    sig = inspect.signature(start_fire_func)
    required_params = sig.parameters.keys()

    start_fire_kwargs = {}
    if 'forest_before_fire' in required_params:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in required_params:
        start_fire_kwargs['neighborhood'] = neighborhood

    wind_directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
    current_wind = wind_direction

    for step in range(M_frames):
        # Zmiana kierunku wiatru co wind_change_interval kroków (nie licząc 0)
        if wind_change_interval and step > 0 and step % wind_change_interval == 0:
            current_wind = random.choice(wind_directions)

        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time = spread_fire_mixed_wind(
                forest_with_fire=forest,
                mixed_forest=mixed_forest,
                p_fire_mixed=p_fire_mixed,
                neighborhood=neighborhood,
                burning_time=burning_time,
                burning_time_dict=burning_time_dict,
                wind_direction=current_wind,
                wind_strength=wind_strength
            )

        fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
        colored_forest = get_colored_forest(forest, mixed_forest)
        ax.imshow(colored_forest)
        ax.axis('off')

        ax.set_title(f'Krok: {step} | Wiatr: {current_wind}, Siła: {wind_strength}')
        ax.legend(handles=legend_elements, loc="lower left", title_fontsize=14, fontsize=12, borderpad=1.0,
                  bbox_to_anchor=(-0.1, -0.1), title="Rodzaje drzew")

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
        dx, dy = wind_arrows.get(current_wind.upper(), (0, 0))

        if dx != 0 or dy != 0:
            ax.annotate(
                '', 
                xy=(0.95, 0.05), 
                xytext=(0.95 - dx * 0.1, 0.05 - dy * 0.1),
                xycoords='axes fraction',
                arrowprops=dict(
                    facecolor='blue', edgecolor='black',
                    linewidth=2, headwidth=10, headlength=10, shrink=0.01
                )
            )
            ax.text(
                0.95, 0.02, 'Wiatr',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10, color='blue',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue')
            )

        plt.draw()
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf)
        frames.append(image)
        plt.close(fig)

    if save:
        frames[0].save(f'{gif_name}.gif', save_all=True, append_images=frames[1:], loop=0, duration=200)
        print(f"Animacja została zapisana jako '{gif_name}.gif'")

fire_simulation_mixed_wind(
    size=100,
    p_tree=0.7,
    tree_ratio={"Łatwopalne": 0.7, "Trudnopalne": 0.3},
    p_fire_mixed={"Łatwopalne": 0.7, "Trudnopalne": 0.3},
    neighborhood='moore',
    burning_time_dict={"Łatwopalne": 1, "Trudnopalne": 1},
    wind_direction="E",
    wind_strength=0.5,
    wind_change_interval=50,
    M_frames=100,
    save=True,
    gif_name="C:/Users/Ala/Desktop/wiatr/wind_change_demo"
)

# wykresy

def simulate_and_measure(size, p_tree, tree_ratio, p_fire_mixed, neighborhood,
                                     burning_time_dict, wind_direction="N", wind_strength=0.2,
                                     start_fire_func=start_fire_claster, M_frames=100):
    forest, mixed_forest = mixed_forest_grid(size, p_tree, tree_ratio=tree_ratio)
    burning_time = np.zeros_like(forest, dtype=int)

    # Uruchomienie pożaru
    sig = inspect.signature(start_fire_func)
    start_fire_kwargs = {}
    if 'forest_before_fire' in sig.parameters:
        start_fire_kwargs['forest_before_fire'] = forest
    if 'neighborhood' in sig.parameters:
        start_fire_kwargs['neighborhood'] = neighborhood

    for step in range(M_frames):
        if step == 1:
            forest = start_fire_func(**start_fire_kwargs)
        elif step > 1:
            forest, burning_time = spread_fire_mixed_wind(
                forest_with_fire=forest,
                mixed_forest=mixed_forest,
                p_fire_mixed=p_fire_mixed,
                neighborhood=neighborhood,
                burning_time=burning_time,
                burning_time_dict=burning_time_dict,
                wind_direction=wind_direction,
                wind_strength=wind_strength
            )

    burned_total = np.sum(forest == tree_cond["BURNED"])
    trees_total = np.sum(forest != tree_cond["EMPTY"])
    return (burned_total / trees_total) * 100 if trees_total > 0 else 0


directions = ["N", "S", "E", "W"]  # Można dodać inne kierunki
strengths = [0.0, 0.3, 0.4, 0.5, 0.6]
repeats = 100  # liczba powtórzeń

results = []

# for direction in directions:
#     row = []
#     for strength in strengths:
#         values = []
#         for _ in range(repeats):
#             burned_percent = simulate_and_measure(
#                 wind_direction=direction,
#                 wind_strength=strength,
#                 size=200,
#                 p_tree=0.7,
#                 tree_ratio={"Łatwopalne": 0.7, "Trudnopalne": 0.3},
#                 p_fire_mixed={"Łatwopalne": 0.7, "Trudnopalne": 0.3},
#                 neighborhood='moore',
#                 burning_time_dict={"Łatwopalne": 2, "Trudnopalne": 3},
#                 M_frames=150
#             )
#             values.append(burned_percent)
#         row.append(np.mean(values))
#     results.append(row)

# sns.heatmap(results, xticklabels=strengths, yticklabels=directions, annot=True, cmap="YlOrRd", fmt=".1f")
# plt.xlabel("Siła wiatru")
# plt.ylabel("Kierunek wiatru")
# plt.title(f"Średni procent spalonego lasu (n={repeats})")
# plt.show()
