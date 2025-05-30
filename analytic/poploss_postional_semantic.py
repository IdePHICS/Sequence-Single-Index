import numpy as np

if __name__ == "__main__":
    from gauss_hermite import GaussHermiteQuadrature
else:
    from .gauss_hermite import GaussHermiteQuadrature

def softmax(x: np.array):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis = -1, keepdims=True)

def teacher(q_teacher: np.array, omega: float, A: np.array):
    return np.add(
        (1-omega) * softmax(np.einsum('...i,...j->...ij', q_teacher, q_teacher)),
        omega * A
    )

def student(q_student: np.array):
    return softmax(np.einsum('...i,...j->...ij', q_student, q_student))
    

def loss(q: np.array, omega: float, A: np.array):
    q_teacher = q[:,:2]
    q_student = q[:,2:]
    return 0.5 * np.linalg.norm((teacher(q_teacher, omega, A) - student(q_student)), axis=(-2,-1,), ord ='fro')**2

def population_loss(e: float, m: float, omega: float, A: np.array, integration_points: int = 10):
    if e**2 + m**2 > 1:
        return np.inf
    mean = np.array([0, 0, e, -e])
    cov = np.array([[1,0,m,0],[0,1,0,m],[m,0,1,0],[0,m,0,1]])
    loss_functional = lambda q: loss(q, omega, A)
    return GaussHermiteQuadrature(loss_functional, mean, cov, n_points=integration_points)


# plot the loss from -1 to 1
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

eps = 1e-6
plot_resolution = 200

def compute_loss_matrix(omega: float, A: np.array, neighborhood_of: tuple = None, resolution: int = plot_resolution, integration_points: int = 17, njobs: int = -1):
    if neighborhood_of is None:
        e_center = 0
        m_center = 0
        e_lower = -1+eps
        e_upper = 1-eps
        m_lower = -1+eps
        m_upper = 1-eps
    else:
        e_center, m_center, neighborhood_radius = neighborhood_of
        e_lower = max(neighborhood_of[0]-neighborhood_radius, -1+eps)
        e_upper = min(neighborhood_of[0]+neighborhood_radius, 1-eps)
        m_lower = max(neighborhood_of[1]-neighborhood_radius, -1+eps)
        m_upper = min(neighborhood_of[1]+neighborhood_radius, 1-eps)

    e, m = np.meshgrid(np.linspace(e_lower, e_upper, resolution), np.linspace(m_lower, m_upper, resolution))

    if njobs != 0:
        loss_matrix = Parallel(n_jobs=njobs)(delayed(population_loss)(x, y, omega, A, integration_points) for x, y in tqdm(zip(e.ravel(), m.ravel()), leave=False))
    else:
        loss_matrix = [population_loss(x, y, omega, A, integration_points) for x, y in tqdm(zip(e.ravel(), m.ravel()), leave=False)]
    loss_matrix = np.array(loss_matrix).reshape(e.shape)
    return loss_matrix, e, m

def surface_minima(X: np.array, Y: np.array, Z: np.array):
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    return X[min_idx], Y[min_idx], Z[min_idx]

def numerical_hessian_initialization(omega: float, A: np.array, resolution: int = plot_resolution, integration_points: int = 17):
    loss, e, m = compute_loss_matrix(omega, A, neighborhood_of=(0, 0, 0.03), resolution=resolution, integration_points=integration_points)

    loss_m, loss_e = np.gradient(loss, m[:,0], e[0,:])

    loss_mm, loss_me = np.gradient(loss_m, m[:,0], e[0,:])
    loss_em, loss_ee = np.gradient(loss_e, m[:,0], e[0,:])

    loss_mm_at_00 = loss_mm[resolution//2, resolution//2]
    loss_ee_at_00 = loss_ee[resolution//2, resolution//2]
    loss_me_at_00 = loss_me[resolution//2, resolution//2]
    loss_em_at_00 = loss_em[resolution//2, resolution//2]

    hessian = np.array([[loss_ee_at_00, loss_me_at_00],
                        [loss_em_at_00, loss_mm_at_00]])
    return hessian

def steepest_direction_initialization(omega: float, A: np.array, resolution: int = plot_resolution, integration_points: int = 17):
    hessian = numerical_hessian_initialization(omega, A, resolution=resolution, integration_points=integration_points)

    # Find lowest eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    min_eigenvalue_index = np.argmin(eigenvalues)
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    min_eigenvector = min_eigenvector / np.linalg.norm(min_eigenvector)
    return min_eigenvector



def loss_3dplot(omega: float, A: np.array, neighborhood_of=None):
    loss_matrix, e, m = compute_loss_matrix(omega, A, neighborhood_of)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(e, m, loss_matrix, cmap='viridis')
    ax.set_xlabel("e")
    ax.set_ylabel("m")
    ax.set_zlabel("loss")
    # plt.show()

def plot_row(omega: float, A: np.array, axes, top=False, neighborhood_of=None):
    if neighborhood_of is None:
        e_center = 0
        m_center = 0
        e_lower = -1+eps
        e_upper = 1-eps
        m_lower = -1+eps
        m_upper = 1-eps
    else:
        e_center, m_center, neighborhood_length = neighborhood_of
        e_lower = max(e_center-neighborhood_length, -1+eps)
        e_upper = min(e_center+neighborhood_length, 1-eps)
        m_lower = max(m_center-neighborhood_length, -1+eps)
        m_upper = min(m_center+neighborhood_length, 1-eps)
    contour_ax, streamplot_ax, gradientnorm_ax, e0section_ax, m0section_ax = axes

    # compute the loss matrix
    loss_matrix, e, m = compute_loss_matrix(omega, A, neighborhood_of)
    

    # find the minimum
    min_idx = np.argmin(loss_matrix)
    min_e, min_m = e.ravel()[min_idx], m.ravel()[min_idx]
    print(f"Minimum at m={min_m}, e={min_e} with omega={omega}")

    # countour plot
    contour_ax.contourf(e, m, loss_matrix, cmap='seismic', levels=100)
    contour_ax.set_xlabel("e")
    contour_ax.set_ylabel("m")
    if e_lower < min_e < e_upper and m_lower < min_m < m_upper:
        contour_ax.scatter(min_e, min_m, color='red', label="Minimum")
    if e_lower < -min_e < e_upper and m_lower < -min_m < m_upper:
        contour_ax.scatter(-min_e, -min_m, color='red')
    if top:
        contour_ax.set_title(f"Contour plot")

    # write a left label with the omega value
    contour_ax.text(-2, 0, f"omega={omega}", ha='center', va='center', rotation=90, fontsize=22)

    # gradientnorm plot
    grad = np.gradient(loss_matrix)
    gradientnorm_ax.contourf(e, m, np.log(grad[0]**2+grad[1]**2), cmap='autumn', levels=100)
    gradientnorm_ax.set_xlabel("e,")
    gradientnorm_ax.set_ylabel("m")
    if top:
        gradientnorm_ax.set_title(f"Gradient plot")

    # streamplot
    streamplot_ax.streamplot(e, m, -grad[1], -grad[0], color=(grad[0]**2+grad[1]**2), cmap='autumn', integration_direction='forward')
    streamplot_ax.set_xlabel("e")
    streamplot_ax.set_ylabel("m")
    if top:
        streamplot_ax.set_title(f"Streamplot")

    # e=e_center section
    ms = np.linspace(m_lower, m_upper, plot_resolution)
    loss_e0 = [population_loss(e_center, m_, omega, A) for m_ in ms]
    e0section_ax.plot(ms, loss_e0)
    e0section_ax.set_xlabel("m")
    e0section_ax.set_ylabel("loss")
    if top:
        e0section_ax.set_title(f"e={e_center} section")

    # m=m_center section
    es = np.linspace(e_lower, e_upper, plot_resolution)
    loss_m0 = [population_loss(e_, m_center, omega, A) for e_ in es]
    m0section_ax.plot(es, loss_m0)
    m0section_ax.set_xlabel("e")
    m0section_ax.set_ylabel("loss")
    if top:
        m0section_ax.set_title(f"m={m_center} section")

if __name__ == "__main__":
    a= .75
    A = softmax(np.array([[a**2, -a**2], [-a**2, a**2]]))
    print(A)

    omegas = [0.1, 0.65, 0.75, .9]
    # omegas = np.linspace(0.3875, 0.4, 5)
    nb = (0,0.45, 0.05)
    nb = None
    # n_omegas = len(omegas)
    # fig, axes = plt.subplots(n_omegas, 5, figsize=(10, 20))
    # for i, omega in enumerate(omegas):
    #     plot_row(omega, A, axes[i], top=(i==0), neighborhood_of=nb)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("loss_diagram.png", dpi=300)


    loss_3dplot(.95, A, neighborhood_of=nb)
    plt.show()



    





