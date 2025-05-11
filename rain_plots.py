import numpy as np
import matplotlib.pyplot as plt
from fire_copy import (
    tree_cond, 
    forest_grid, 
    start_fire, 
    spread_fire,
    generate_rain_mask)


def how_much_burned(size, p_fire, p_tree, neighborhood, M=100, burn_time=1, rain_intensity=0, rain_mask=None, wet_timer=None, wet_time=3):
    burned = np.zeros(M)
    forest = forest_grid(size, p_tree)
    total_trees = np.sum(forest == tree_cond["TREE"])
    forest = start_fire(forest)
    burn_timer = np.zeros_like(forest, dtype=int)
    wet_timer = np.zeros_like(forest, dtype=int)
    
    for step in range(M):
        rain_mask = generate_rain_mask(size, rain_type="random", step=step, n_clouds=3, cloud_size=600, dx=-1, dy=1)
        burned_trees = np.sum(forest == tree_cond["BURNED"]) + np.sum(forest == tree_cond["BURNING"])
        burned[step] = (burned_trees / total_trees) * 100 if total_trees > 0 else 0
        forest, burn_timer, wet_timer = spread_fire(forest, burn_timer, p_fire, neighborhood, burn_time, rain_intensity, rain_mask, wet_timer, wet_time)

    return burned, np.arange(M)


N = 30
M = 200
p_fire, p_tree = 1, 0.6
size = 50
burn_time = 1
wet_time = 10
burned_rain_small = np.zeros((N, M))
burned_rain_medium = np.zeros((N, M))
burned_rain_big = np.zeros((N, M))
steps = np.arange(M)

for i in range(N):
    burned_rain_small[i,:], _ = how_much_burned(size, p_fire=p_fire, p_tree=p_tree, neighborhood="moore", M=M, burn_time=burn_time, rain_intensity=0.2, rain_mask=None, wet_timer=None, wet_time=wet_time)
    burned_rain_medium[i,:], _ = how_much_burned(size, p_fire=p_fire, p_tree=p_tree, neighborhood="moore", M=M, burn_time=burn_time, rain_intensity=0.5, rain_mask=None, wet_timer=None, wet_time=wet_time)
    burned_rain_big[i,:], _ = how_much_burned(size, p_fire=p_fire, p_tree=p_tree, neighborhood="moore", M=M, burn_time=burn_time, rain_intensity=0.9, rain_mask=None, wet_timer=None, wet_time=wet_time)


avg_burned_small = np.mean(burned_rain_small, axis=0)
avg_burned_medium = np.mean(burned_rain_medium, axis=0)
avg_burned_big = np.mean(burned_rain_big, axis=0)

plt.figure(figsize=(10, 7))
plt.plot(steps, avg_burned_small, label="rain_intensity = 0.2", color="blue")
plt.plot(steps, avg_burned_medium, label="rain_intensity = 0.5", color="orange")
plt.plot(steps, avg_burned_big, label="rain_intensity = 0.9", color="red")
plt.xlabel('Krok symulacji')
plt.ylabel('Średni rozmiar spalonego obszaru (%)')
plt.title(f"p_tree = {p_tree}, p_fire = {p_fire}, grid_size = {size}x{size}, burn_time = {burn_time}, wet_time = {wet_time}")
plt.legend(loc = "lower right")
plt.grid(True, linestyle = '--')
plt.tight_layout()
#plt.savefig("avg100_burned_vs_steps.png", dpi=300)
plt.show()