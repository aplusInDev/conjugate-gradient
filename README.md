# Conjugate Gradient

This repository contains the implementation of the Conjugate Gradient method, a popular algorithm for solving systems of linear equations with a symmetric positive-definite matrix.

## Authors

- [**AplusInDev**](https://www.github.com/AplusInDev)

## Prerequisites

- Python 3.6 or higher
- pip
- git

## Installation

### Windows

1. Clone the repository

    ```bash
    git clone https://www.github.com/AplusInDev/conjugate-gradient
    ```

2. Change the working directory

    ```bash
    cd conjugate-gradient
    ```

3. Create a virtual environment

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment

    ```bash
    .venv\Scripts\activate
    ```

5. Install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

### Linux/MacOS

1. Clone the repository

    ```bash
    git clone https://www.github.com/AplusInDev/conjugate-gradient
    ```

2. Change the working directory

    ```bash
    cd conjugate-gradient
    ```

3. Create a virtual environment

    ```bash
    python -m venv .venv
    ```

4. Activate the virtual environment

    ```bash
    source .venv/bin/activate
    ```

5. Install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```python
import numpy as np
from utils import conjugate_gradient

# Define the matrix A and the vector b
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])

# Solve the system of linear equations
x0, k, x_values = conjugate_gradient(A, b)

print(f"Solution: {x0}, Iterations: {k}, x_values: {x_values}")
```

## Example

**try:**

```bash
python index.py
```
