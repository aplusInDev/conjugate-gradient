""" Main file to run the conjugate gradient method. """

import numpy as np
from typing import List, Union
from utils import (
    conjugate_gradient,
    animate_conjugate_gradient,
    animate_conjugate_gradient_3d
)


A: List[List[Union[int, float]]] = np.array([[4, 1], [1, 3]])
b: List[Union[int, float]] = np.array([1, 2])
x0: List[Union[int, float]] = np.array([0, 0])

def main():
    """ Main function to run the conjugate gradient method. """
    A = np.array([[4, 4], [4, 6]])
    b = np.array([5, 6])
    # A = np.array([[4, 1], [1, 3]])
    # b = np.array([1, 2])
    x0 = np.array([0, 0])

    _, _, x_values = conjugate_gradient(A, b, x0)
    # print(x_values)
    animate_conjugate_gradient(A, b, x_values)
    # animate_conjugate_gradient_3d(A, b, x_values)


if __name__ == '__main__':
    main()
