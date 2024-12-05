""" Utilities for the Conjugate Gradient Method
This module provides a function to solve a system of linear equations using
the conjugate gradient method and another function to animate the iterations
of the conjugate gradient method.
Conjugate Gradient Method
The algorithm is as follows:
1. Initialize x0, r0, p0
2. Iterate until the residual is small enough:
    a. Calculate αk = (r_k, r_k) / (p_k, A p_k)
    b. Update x_k = x_{k-1} + α_k p_k
    c. Calculate r_k = r_{k-1} - α_k A p_k
    d. Calculate β_k = (r_k, r_k) / (r_{k-1}, r_{k-1})
    e. Update p_k = r_k + β_k p_{k-1}
    f. Update k = k + 1
3. Return x_k
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Union, List, Tuple
from numpy.typing import NDArray


def conjugate_gradient(
    A: List[List[Union[int, float]]],
    b: List[Union[int, float]], 
    x0: Optional[List[Union[int, float]]] = None, 
    e: float = 1e-10
) -> Tuple[List[Union[int, float]], int, List[Union[int, float]]]:
    """
    Solve the system of linear equations Ax = b using the conjugate gradient method.

    The conjugate gradient method is an iterative algorithm for solving 
    symmetric, positive-definite linear systems efficiently.

    Args:
        A (List[List[Union[int, float]]]): Symmetric, positive-definite coefficient matrix
        b (List[Union[int, float]]): Right-hand side vector of the linear system
        x0 (Optional[List[Union[int, float]]], optional): Initial guess vector. 
            Defaults to a zero vector if not provided.
        e (float, optional): Convergence tolerance. Defaults to 1e-10.

    Returns:
        Tuple containing:
        - Solution vector x
        - Number of iterations
        - List of intermediate solution vectors

    Raises:
        ValueError: If the input matrix is not symmetric or positive-definite

    Notes:
        - Requires numpy for numerical operations

    Example:
        >>> A = [[4, 2], [2, 6]]
        >>> b = [5, 6]
        >>> conjugate_gradient(A, b)
        ([1.0, 1.0], 2, [[0.0, 0.0], [1.0, 1.0]])
        >>> ## Example of not symmetric matrix
        >>> A = [[4, 1], [2, 6]]
        >>> b = [5, 6]
        >>> conjugate_gradient(A, b)
        ValueError: Matrix A must be symmetric
        >>> ## Example of not positive-definite matrix
        >>> A = [[-4, 2], [2, 6]]
        >>> b = [5, 6]
        >>> conjugate_gradient(A, b)
        ValueError: Matrix A must be positive-definite
    """
    ## raise ValueError if the input matrix is not symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")
    ## raise ValueError if the input matrix is not positive-definite
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix A must be positive-definite")
    ## initial guess
    x0 = np.zeros_like(b) if x0 is None else np.array(x0)
    ## initial residual, calculated as b - Ax0
    r0 = b - np.dot(A, x0)
    ## initial search direction
    d0 = r0
    ## iteration counter
    k = 0

    x_values = [x0.copy()]
    ## iterate until the residual is small enough
    while np.linalg.norm(r0) > e:
        αk = np.dot(r0, r0) / np.dot(d0, np.dot(A, d0))
        x0 = x0 + αk * d0
        rk = r0 - αk * np.dot(A, d0)
        βk = np.dot(rk, rk) / np.dot(r0, r0)
        d0 = rk + βk * d0
        r0 = rk
        k += 1
        x_values.append(x0.copy())
        # print(f"Iteration {k}: x = {x0}, residual = {r0}, direction = {d0}, alpha = {αk}, beta = {βk}")

    x_values = np.array(x_values)
    
    return x0, k, x_values


def animate_conjugate_gradient(
    A: Union[List[List[Union[int, float]]], NDArray[np.float64]],
    b: Union[List[Union[int, float]], NDArray[np.float64]],
    x_values: Union[List[List[Union[int, float]]], NDArray[np.float64]]
) -> None:
    """
    Visualize the conjugate gradient method's iterative solution convergence.

    This function creates an animated contour plot showing how the solution 
    iteratively approaches the optimal point in the solution space.

    Args:
        A (List[List[Union[int, float]]]): Symmetric, positive-definite coefficient matrix
        b (List[Union[int, float]]): Right-hand side vector of the linear system
        x_values (List[Union[int, float]]): Sequence of intermediate solution vectors

    Raises:
        ValueError: If input matrix A is not symmetric or positive-definite

    Side Effects:
        - Generates an animated matplotlib figure
        - Displays the convergence path of the conjugate gradient method

    Notes:
        - Requires matplotlib and numpy for visualization
        - Animation shows gradual convergence of solution vector
        - Contour plot represents the objective function landscape

    Example:
        >>> A = [[4, 2], [2, 6]]
        >>> b = [5, 6]
        >>> x_values = [[0.0, 0.0], [1.0, 1.0]]
        >>> animate_conjugate_gradient(A, b, x_values)
    """

    ## Raise ValueError if the input matrix is not symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")
    ## Raise ValueError if the input matrix is not positive-definite
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix A must be positive-definite")

    x = np.linspace(-1, 1, 800)
    y = np.linspace(-1, 1, 800)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = 0.5 * np.dot([X[i, j], Y[i, j]], np.dot(A, [X[i, j], Y[i, j]])) - np.dot(b, [X[i, j], Y[i, j]])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contour(X, Y, Z, levels=20)
    line, = ax.plot([], [], marker='o', linestyle='-', color='b', markersize=8, linewidth=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x_values[:i+1, 0], x_values[:i+1, 1])
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_values), interval=500, blit=True, repeat=False)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Conjugate Gradient Method Iterations')
    plt.grid(True)
    plt.show()

def animate_conjugate_gradient_3d(
    A: Union[List[List[Union[int, float]]], NDArray[np.float64]],
    b: Union[List[Union[int, float]], NDArray[np.float64]],
    x_values: Union[List[List[Union[int, float]]], NDArray[np.float64]]
) -> None:
    """
    Visualize the conjugate gradient method's iterative solution convergence in 3D.

    This function creates an animated 3D surface plot showing how the solution 
    iteratively approaches the optimal point in the solution space.

    Args:
        A (List[List[Union[int, float]]]): Symmetric, positive-definite coefficient matrix
        b (List[Union[int, float]]): Right-hand side vector of the linear system
        x_values (List[Union[int, float]]): Sequence of intermediate solution vectors

    Raises:
        ValueError: If input matrix A is not symmetric or positive-definite

    Side Effects:
        - Generates an animated matplotlib figure
        - Displays the convergence path of the conjugate gradient method

    Notes:
        - Requires matplotlib and numpy for visualization
        - Animation shows gradual convergence of solution vector
        - 3D surface plot represents the objective function landscape

    Example:
        >>> A = [[4, 2], [2, 6]]
        >>> b = [5, 6]
        >>> x_values = [[0.0, 0.0], [1.0, 1.0]]
        >>> animate_conjugate_gradient_3d(A, b, x_values)
    """

    ## Raise ValueError if the input matrix is not symmetric
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")
    ## Raise ValueError if the input matrix is not positive-definite
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix A must be positive-definite")

    x = np.linspace(-1, 1, 800)
    y = np.linspace(-1, 1, 800)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = 0.5 * np.dot([X[i, j], Y[i, j]], np.dot(A, [X[i, j], Y[i, j]])) - np.dot(b, [X[i, j], Y[i, j]])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

    line, = ax.plot([], [], [], marker='o', linestyle='-', color='r', markersize=8, linewidth=2)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def animate(i):
        line.set_data(x_values[:i+1, 0], x_values[:i+1, 1])
        line.set_3d_properties([0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x) for x in x_values[:i+1]])
        return line,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_values), interval=500, blit=True, repeat=False)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Function Value')
    ax.set_title('Conjugate Gradient Method Iterations')

    plt.show()
