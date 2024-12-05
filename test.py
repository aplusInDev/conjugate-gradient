import unittest
import numpy as np
from utils import conjugate_gradient

class CustomTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"✅ {test._testMethodName} pass.")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"❌ {test._testMethodName} failed.")

class CustomTestRunner(unittest.TextTestRunner):
    resultclass = CustomTestResult

class TestConjugateGradient(unittest.TestCase):

    def test_conjugate_gradient(self):
        A = np.array([[4, 2], [2, 6]])
        b = np.array([5, 6])
        x, iterations, x_values = conjugate_gradient(A, b)
        # print(f"Solution: {x}")
        # print(f"Iterations: {iterations}")
        # print(f"Intermediate solutions: {x_values}")
        np.testing.assert_almost_equal(x, [0.9, 0.7], decimal=5)
        self.assertEqual(iterations, 2)
        np.testing.assert_almost_equal(x_values[-1], [0.9, 0.7], decimal=5)

    def test_non_symmetric_matrix(self):
        A = np.array([[4, 1], [2, 6]])
        b = np.array([5, 6])
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b)

    def test_non_positive_definite_matrix(self):
        A = np.array([[-4, 2], [2, 6]])
        b = np.array([5, 6])
        with self.assertRaises(ValueError):
            conjugate_gradient(A, b)


if __name__ == '__main__':
    unittest.main(testRunner=CustomTestRunner())
