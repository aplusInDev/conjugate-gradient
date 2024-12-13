# Conjugate Gradient Method

## Table of Contents

1. [Overview](#overview)
2. [Authors](#authors)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
    1. [Windows](#windows-installation)
    2. [Linux/MacOS](#linuxmacos-installation)
5. [Usage](#usage)
6. [Example](#example)

## Overview

This repository contains the implementation of the Conjugate Gradient method, a popular algorithm for solving systems of linear equations with a symmetric positive-definite matrix.

## Authors

- [**AplusInDev**](https://www.github.com/AplusInDev)

## Prerequisites

- Python 3.6 or higher
- pip
- git

## Installation

### Windows Installation

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

### Linux/MacOS Installation

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

**try:**

```bash
python index.py
```

## Example

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

**Output:**
```
Solution: [0.21052632 0.57894737], Iterations: 2, x_values: [array([0.5, 0.5]), array([0.21052632, 0.57894737])]
```
