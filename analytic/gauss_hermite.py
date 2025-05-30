# Implement a multi-dimensional Gauss-Hermite quadrature rule. 

import numpy as np
from scipy.special import roots_hermitenorm
# typying for numpy array
from typing import Callable




def GaussHermiteQuadrature(f: Callable, mean: np.ndarray, cov: np.ndarray, n_points: int = 3, method = 'cholesky') -> float:
    """
    Compute the Gauss-Hermite quadrature rule for a multivariate Gaussian distribution.

    Parameters:
    mean (np.ndarray): The mean of the Gaussian distribution.
    cov (np.ndarray): The covariance matrix of the Gaussian distribution.
    n_points (int): The number of quadrature points to compute.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the quadrature points and weights.
    """
    # Compute the point traformation matrix
    if method == 'svd':
        U, S, _ = np.linalg.svd(cov)
        L = np.dot(U, np.diag(np.sqrt(S)))
    elif method == 'cholesky':
        L = np.linalg.cholesky(cov)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    # Compute the roots of the Hermite polynomial
    x, w = roots_hermitenorm(n_points)
    
    # Compute the quadrature points
    plain_pointgrid = np.meshgrid(*[x]*mean.shape[0])
    plain_pointgrid = np.column_stack([coordinate.ravel() for coordinate in plain_pointgrid])
    shifted_pointgrid = np.dot(plain_pointgrid, L.T) + mean
    
    # Compute the quadrature weights
    weightgrid = np.meshgrid(*[w]*mean.shape[0])
    weightgrid = np.column_stack([coordinate.ravel() for coordinate in weightgrid])
    weightgrid = np.prod(weightgrid, axis=1)
    

    integral = np.sum(f(shifted_pointgrid) * weightgrid) 
    integral /= np.sqrt(2*np.pi)**mean.shape[0] # Normalization factor


    return integral



if __name__ == "__main__":
    # Define the mean and covariance of the Gaussian distribution
    size = 2
    mean = np.array([1., -2.])
    cov = np.array([[1., 0.5], [0.5, 1.]])
    
    # Define the function to integrate

    f = lambda x: np.tanh(x[:, 0]+x[:, 1])

    # multivariate gaussian pdf with scipy
    from scipy.stats import multivariate_normal
    f_scipy = lambda x,y: np.tanh(x+y) * multivariate_normal.pdf([x,y], mean=mean, cov=cov)

    # Compute the Gauss-Hermite quadrature rule with svd
    n_points = list(range(1, 50))
    svd_integral = [GaussHermiteQuadrature(f, mean, cov, n) for n in n_points]
    cholesky_integral = [GaussHermiteQuadrature(f, mean, cov, n, method='cholesky') for n in n_points]

    print(svd_integral[-1])
    print(cholesky_integral[-1])

    # compare with scipy quadrature
    from scipy.integrate import dblquad
    result = dblquad(f_scipy, -np.inf, np.inf, -np.inf, np.inf)
    print(result[0], '±', result[1])
    
    # plot 
    import matplotlib.pyplot as plt
    plt.plot(n_points, svd_integral, label="SVD")
    plt.plot(n_points, cholesky_integral, label="Cholesky")
    plt.axhline(result[0], color="black", linestyle="--", label="Scipy")
    plt.xlabel("Number of quadrature points")
    plt.ylabel("Integral value")
    plt.legend()
    plt.show()


    

