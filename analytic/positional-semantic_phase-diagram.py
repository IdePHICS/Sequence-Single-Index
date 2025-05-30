import numpy as np
from tqdm import tqdm
from plotstyle import PlotStyle

from poploss_postional_semantic import compute_loss_matrix, population_loss, steepest_direction_initialization

np.set_printoptions(linewidth=500, threshold=100) # for debugging

# Best fast resolution
resolution = 200
integration_points = 19
a_grid, omega_grid = np.meshgrid(
    np.linspace(0.7, 1, resolution),
    np.linspace(0.4, 1, resolution),
)
fast = True

# Best complete resolution
# resolution = 100
# integration_points = 20
# a_grid, omega_grid = np.meshgrid(
#     np.linspace(0, 1, resolution),
#     np.linspace(0, 1, resolution),
# )
# fast = False

a_space = a_grid[0,:]
omega_space = omega_grid[:,0]
import os
loss_cachedir = "analytic-database/positional-semantic_phase-diagram/loss_cache"
minimas_cachedir = "analytic-database/positional-semantic_phase-diagram/loss-minimas_cache"
diagram_cachedir = "analytic-database/positional-semantic_phase-diagram/diagram_cache"
steepest_cachedir = "analytic-database/positional-semantic_phase-diagram/steepest_cache"
for dir in [loss_cachedir, minimas_cachedir, diagram_cachedir, steepest_cachedir]:
    os.makedirs(dir, exist_ok=True)

eps = 1e-10

## NOT USED ##
def is_semantic_global_minima(A: np.array, omega: float) -> bool:
    return population_loss(0, 1.-eps, omega, A) > population_loss(1, 0, omega, A) 

def is_positional_global_minima(A: np.array, omega: float) -> bool:
    return not is_semantic_global_minima(A, omega)

def towards_semantic(A: np.array, omega: float) -> bool:
    steepest_direction = steepest_direction_initialization(omega, A)
    return abs(np.dot(np.array([0,1]), steepest_direction)) > abs(np.dot(np.array([1,0]), steepest_direction))

def towards_positional(A: np.array, omega: float) -> bool:
    return not towards_semantic(A, omega)
##############

def is_semantic_local_minima(A: np.array, omega: float) -> bool:
    neighborhood = (0, 1, 2.5/resolution)
    nb_loss, nb_e, nb_m = compute_loss_matrix(omega, A, neighborhood_of=neighborhood, resolution=10, integration_points=integration_points)
    nb_loss -= population_loss(0, 1.-eps, omega, A, integration_points=integration_points)
    return (nb_loss >= 0).all()

def is_positional_local_minima(A: np.array, omega: float) -> bool:
    neighborhood = (1, 0, 2.5/resolution)
    nb_loss, nb_e, nb_m = compute_loss_matrix(omega, A, neighborhood_of=neighborhood, resolution=10, integration_points=integration_points)
    nb_loss -= population_loss(1, 0, omega, A, integration_points=integration_points)
    return (nb_loss >= 0).all()

from scipy.special import softmax
def a_to_A(a: float) -> np.array:
    """
    Convert a to A
    """
    return softmax(np.array([[a** 2, -a**2], [-a**2, a**2]]), axis = -1)


def cached_compute_loss_matrix(a: float, omega: float, no_compute: bool = False, no_cache: bool = False) -> tuple:
    """
    Compute the loss matrix and save it to a file
    """
    filename = f"{loss_cachedir}/a{a:.4f}_omega{omega:.4f}_resolution{resolution}_integrationpoints{integration_points}.npz"
    if os.path.exists(filename) and not no_cache:
        data = np.load(filename)
        loss = data["loss"]
        e = data["e"]
        m = data["m"]
    else:
        if no_compute:
            raise ValueError(f"no_compute is True, but file does not exist: a={a}, omega={omega}, res={resolution}, integration_points={integration_points}.")
        loss, e, m = compute_loss_matrix(omega, a_to_A(a), resolution=resolution, integration_points=integration_points)
        np.savez(filename, loss=loss, e=e, m=m)
    return loss, e, m


def first_quadrant_local_minimas_guess(loss,e, m):
    # pad loss with inf
    padded_loss = np.pad(loss, ((1, 1), (1, 1)), mode='constant', constant_values=+np.inf)
    # print("loss\n", loss)
    # print("padded_loss\n", padded_loss)


    is_local_minima = np.zeros_like(loss, dtype=bool)
    # 8 directions shifts
    left_top = padded_loss[:-2, :-2]
    center_top = padded_loss[1:-1, :-2]
    right_top = padded_loss[2:, :-2]
    left_center = padded_loss[:-2, 1:-1]
    right_center = padded_loss[2:, 1:-1]
    left_bottom = padded_loss[:-2, 2:]
    center_bottom = padded_loss[1:-1, 2:]
    right_bottom = padded_loss[2:, 2:]
    # check if the stencil catch at lest one direction
    is_good_minima_computation = np.logical_or(
        np.logical_or(
            np.logical_and(left_top != np.inf, right_bottom != np.inf),
            np.logical_and(center_top != np.inf, center_bottom != np.inf)
        ),
        np.logical_or(
            np.logical_and(left_center != np.inf, right_center != np.inf),
            np.logical_and(left_bottom != np.inf, right_top != np.inf)
        )
    )
    # check if the loss is lower than all 8 directions
    is_local_minima = np.logical_and(
        loss != np.inf,
        np.logical_and(
            np.logical_and(
                np.logical_and(loss <= left_top, loss <= center_top),
                np.logical_and(loss <= right_top, loss <= left_center)
            ),
            np.logical_and(
                np.logical_and(loss <= right_center, loss <= left_bottom),
                np.logical_and(loss <= center_bottom, loss <= right_bottom)
            )
        )
    )

    is_first_quadrant = np.logical_and(e >= 0, m >= 0)
    good_minima = np.logical_and(
        is_local_minima,
        np.logical_and(
            is_good_minima_computation,
            is_first_quadrant
        )
    )

    # find the global minima of loss with argmin
    first_quadrant_indices = np.where(is_first_quadrant)
    minima_index = np.argmin(loss[first_quadrant_indices])
    e_min = e[first_quadrant_indices][minima_index]
    m_min = m[first_quadrant_indices][minima_index]
    first_quadrant_local_minima = np.logical_and(e == e_min, m == m_min)
    good_minima = np.logical_or(good_minima, first_quadrant_local_minima)

    e_minima = e[good_minima]
    m_minima = m[good_minima]
    loss_minima = loss[good_minima]

    # minima_stencil = np.array([left_top[good_minima], center_top[good_minima], right_top[good_minima], left_center[good_minima], loss[good_minima], right_center[good_minima], left_bottom[good_minima], center_bottom[good_minima], right_bottom[good_minima]]).T.reshape(-1, 3,3)
    # print("minima_stencil:\n", minima_stencil)


    return e_minima, m_minima, loss_minima

def augment_local_minimas(a: float, omega: float, e_minima: np.array, m_minima: np.array, loss_minima: np.array):
    radius = 2.5/resolution

    # check of precise extremes
    exact_semantic_minima = False
    if is_semantic_local_minima(a_to_A(a), omega):
        exact_semantic_minima = True

    exact_positional_minima = False
    if is_positional_local_minima(a_to_A(a), omega):
        exact_positional_minima = True
    

    # remove approximated extremes
    verified = np.zeros_like(loss_minima, dtype=bool)
    for index, (em, mm) in enumerate(zip(e_minima, m_minima)):
        if (em >= 1-radius-3*eps and exact_positional_minima) or (mm >= 1-radius-3*eps and exact_semantic_minima):
            verified[index] = False
        else:
            verified[index] = True
    
    e_minima = e_minima[verified]
    m_minima = m_minima[verified]
    loss_minima = loss_minima[verified]

    if exact_positional_minima:
        e_minima = np.append(e_minima, 1.)
        m_minima = np.append(m_minima, 0.)
        loss_minima = np.append(loss_minima, population_loss(1, 0, omega, a_to_A(a), integration_points=integration_points))
    if exact_semantic_minima:
        e_minima = np.append(e_minima, 0.)
        m_minima = np.append(m_minima, 1.)
        loss_minima = np.append(loss_minima, population_loss(0, 1.-eps, omega, a_to_A(a), integration_points=integration_points))

    return e_minima, m_minima, loss_minima


def cache_compute_minimas(a: float, omega: float, no_compute: bool = False, no_cache: bool = False, fast: bool = False):
    """
    Compute the local minimas in the first quadrant
    """
    filename = f"{minimas_cachedir}/a{a:.4f}_omega{omega:.4f}_resolution{resolution}_integrationpoints{integration_points}{'_fast' if fast else ''}.npz"
    if os.path.exists(filename) and not no_cache:
        data = np.load(filename)
        e_minima = data["e_minima"]
        m_minima = data["m_minima"]
        loss_minima = data["loss_minima"]
    else:
        if no_compute:
            raise ValueError(f"no_compute is True, but file does not exist: a={a}, omega={omega}, res={resolution}, integration_points={integration_points}.")
        if fast:
            e_minima = np.array([])
            m_minima = np.array([])
            loss_minima = np.array([])
        else:
            loss, e, m = cached_compute_loss_matrix(a, omega, no_compute=no_compute)
            e_minima, m_minima, loss_minima = first_quadrant_local_minimas_guess(loss, e, m)
        e_minima, m_minima, loss_minima = augment_local_minimas(a, omega, e_minima, m_minima, loss_minima)

        # Fallback if using fast mode and no minimas are found
        if fast and len(e_minima) == 0:
            print(f"Fast mode: no minimas found for a={a}, omega={omega}. Computing all minimas.")
            loss, e, m = cached_compute_loss_matrix(a, omega, no_compute=no_compute)
            e_minima, m_minima, loss_minima = first_quadrant_local_minimas_guess(loss, e, m)
            e_minima, m_minima, loss_minima = augment_local_minimas(a, omega, e_minima, m_minima, loss_minima)
        np.savez(filename, e_minima=e_minima, m_minima=m_minima, loss_minima=loss_minima)
    return e_minima, m_minima, loss_minima

def cache_compute_steepest_direction(a: float, omega: float, no_cache: bool = False, no_compute: bool = False):
    """
    Compute the steepest direction
    """
    filename = f"{steepest_cachedir}/a{a:.4f}_omega{omega:.4f}_resolution{resolution}_integrationpoints{integration_points}.npz"
    if os.path.exists(filename) and not no_cache:
        data = np.load(filename)
        steepest_direction = data["steepest_direction"]
    else:
        if no_compute:
            raise ValueError(f"no_compute is True, but file does not exist: a={a}, omega={omega}, res={resolution}, integration_points={integration_points}.")
        steepest_direction = steepest_direction_initialization(omega, a_to_A(a), resolution=min(resolution,10), integration_points=integration_points)
        np.savez(filename, steepest_direction=steepest_direction)
    return steepest_direction

def global_minima(a: float, omega: float, fast: bool = False):
    """
    Compute the global minimas
    """
    minimas = cache_compute_minimas(a, omega, fast=fast)
    try:
        index = np.argmin(minimas[2])
    except ValueError:
        raise ValueError(f"Unable to find the minimum at a={a}, omega={omega}.")
    return minimas[0][index], minimas[1][index], minimas[2][index]

def positional_global_minima(a: float, omega: float, fast: bool = False):
    """
    Compute the global minimas
    """
    e_minima, m_minima, loss_minima = global_minima(a, omega, fast=fast)
    return e_minima

def positional_dynamic(a: float, omega: float):
    """
    Compute the dynamic of the loss
    """
    steepest_direction = cache_compute_steepest_direction(a, omega)

    return abs(steepest_direction[0])

def multiple_minimas(a: float, omega: float, fast: bool = False):
    """
    Count the number of minimas
    """
    e_minima, m_minima, loss_minima = cache_compute_minimas(a, omega, fast=fast)
    return float(len(e_minima) > 1)

def cache_compute_diagram(no_compute: bool = False, no_cache: bool = False, fast: bool = False):
    """
    Compute the diagram
    """
    filename = f"{diagram_cachedir}/resolution{resolution}_integrationpoints{integration_points}{'_fast' if fast else ''}.npz"
    if os.path.exists(filename) and not no_cache:
        data = np.load(filename)
        positional_global_minima_grid = data["positional_global_minima_grid"]
        positional_dynamic_grid = data["positional_dynamic_grid"]
        multiple_minimas_grid = data["multiple_minimas_grid"]
    else:
        if no_compute:
            raise ValueError(f"no_compute is True, but file does not exist: res={resolution}, integration_points={integration_points}.")
        
        positional_global_minima_grid = np.zeros_like(a_grid).ravel()
        positional_dynamic_grid = np.zeros_like(a_grid).ravel()
        multiple_minimas_grid = np.zeros_like(a_grid).ravel()
        for i, (a, omega) in tqdm(enumerate(zip(a_grid.ravel(), omega_grid.ravel())), total=resolution**2):
            positional_global_minima_grid[i] = positional_global_minima(a, omega, fast=fast)
            positional_dynamic_grid[i] = positional_dynamic(a, omega)
            multiple_minimas_grid[i] = multiple_minimas(a, omega, fast=fast)
        positional_global_minima_grid = positional_global_minima_grid.reshape(a_grid.shape)
        positional_dynamic_grid = positional_dynamic_grid.reshape(a_grid.shape)
        multiple_minimas_grid = multiple_minimas_grid.reshape(a_grid.shape)
        
        np.savez(
            filename,
            positional_global_minima_grid=positional_global_minima_grid,
            positional_dynamic_grid=positional_dynamic_grid,
            multiple_minimas_grid=multiple_minimas_grid
        )

    return positional_global_minima_grid, positional_dynamic_grid, multiple_minimas_grid

if __name__ == "__main__":
    # for a, omega in tqdm(zip(a_grid.ravel(), omega_grid.ravel()), total=resolution**2):
    #     _ = cached_compute_loss_matrix(a, omega)


    positional_global_minima_grid, positional_dynamic_grid, multiple_minimas_grid = cache_compute_diagram(fast=fast)

   # make 3 heatrmap plots of the diagram
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Phase diagram of the positional-semantic loss")
    ax[0].pcolor(a_grid, omega_grid, positional_global_minima_grid, shading='auto', vmin=0, vmax=1, cmap='winter')
    ax[0].set_title("Positional global minima")
    ax[0].set_xlabel("a")
    ax[0].set_ylabel("omega")
    ax[1].pcolor(a_grid, omega_grid, positional_dynamic_grid, shading='auto', vmin=0, vmax=1, cmap='winter')
    ax[1].set_title("Positional dynamic")
    ax[1].set_xlabel("a")
    ax[1].set_ylabel("omega")
    ax[2].pcolor(a_grid, omega_grid, multiple_minimas_grid, shading='auto', vmin=0, vmax=1, cmap='winter')
    ax[2].set_title("Multiple minimas")
    ax[2].set_xlabel("a")
    ax[2].set_ylabel("omega")
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    # show only one colorbar for all plots with vmin=0 and vmax=1. From blue to red
    cbar = fig.colorbar(ax[0].collections[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    plt.savefig("phase_diagram.png", dpi=300)
    # plt.show()

    global_minima = np.heaviside(positional_global_minima_grid-.5,0)
    dynamic_direction = np.heaviside(positional_dynamic_grid-.5,0)
    multiple_minimas = np.heaviside(multiple_minimas_grid-.5,0)

    combination = (global_minima * 1 + dynamic_direction * 2 + multiple_minimas * 4).astype(int)
    regions = dict({
        ## [global][dynamic][multiple]
        # 000
        'Unique Semantic Minima': '#339966', # green
        # 100
        'SKIP Unique Positional Minima, Misaligned Dynamic': '#ff9966', # orange
        # 010
        'Unique Semantic Minima, Misaligned Dynamic': '#ffcc00', # yellow
        # 110
        'Unique Positional Minima': '#3333ff', # blue
        # 001
        'Global Semantic Minima': '#66ff33', # light green
        # 101
        'SKIP Global Positional Minima, Semantic Dynamic': '#cc3300', # brown
        # 011
        'Global Semantic Minima, Positional Dynamic': '#ff5050', # red
        # 111
        'Global Positional Minima': '#99ccff', # light blue

    })

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(colors=list(regions.values()))

    with PlotStyle():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(a_grid, omega_grid, combination, cmap=custom_cmap, shading='auto')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=regions[key], label=key)  for key in regions.keys() if not key.startswith('SKIP')]
        ax.legend(handles=legend_elements, loc='lower left')

        ax.set_xlim(min(a_space), max(a_space))
        ax.set_ylim(min(omega_space), max(omega_space))
        ax.set_xlabel("$a$")
        ax.set_ylabel("$\omega$")
        # ax.set_title("Phase diagram of the positional-semantic loss")
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig.savefig("figures/positional-semantic/phase_diagram.pdf", dpi=300)
        plt.show()




    a = a_space[-1]
    omega = omega_space[-1]
    print(f"a: {a}, omega: {omega}")
    loss, e, m = cached_compute_loss_matrix(a, omega, no_compute=True)
    minima = cache_compute_minimas(a, omega, no_cache=True)
    print(f"minima\n: {minima}")

    # print(f"positional global minima: {positional_global_minima(a, omega)}")
    print(f"positional dynamic: {positional_dynamic(a, omega)}")
    import matplotlib.pyplot as plt

    #3d plot of the loss
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(e, m, loss, cmap='viridis', linewidth=0, antialiased=True, zorder=1)
    ax.set_xlabel("e")
    ax.set_ylabel("m")
    ax.set_zlabel("loss")
    ax.plot(minima[0], minima[1], minima[2], 'ro', markersize=5, label="local minimas", zorder=2)
    plt.show()



