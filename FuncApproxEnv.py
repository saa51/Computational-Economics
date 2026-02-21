from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import time

from FunctionApprox import ChebyshevApprox, PolynomialApprox, SplineApprox

default_params: Dict[str, float] = {"alpha": 0.3, "beta": 0.98, "upper": 10, "lower": 0}


class FuncApproxEnv:
    def __init__(self, params: Optional[Dict[str, float]] = None) -> None:
        if params is None:
            params = {}
        self.alpha = params.get("alpha", default_params["alpha"])
        self.beta = params.get("beta", default_params["beta"])
        self.upper = params.get("upper", default_params["upper"])
        self.lower = params.get("lower", default_params["lower"])

    def real_func(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * self.beta * x**self.alpha

    def validate(self, n: int, grid_num: int = 100, plot: bool = True) -> None:
        test_grids = np.linspace(self.lower, self.upper, grid_num, endpoint=True)
        real_value = self.real_func(test_grids)

        even_grids = np.linspace(self.lower, self.upper, n, endpoint=True)
        even_value = self.real_func(even_grids)

        start_time = time.time()
        lin = SplineApprox()
        lin.approx(even_grids, even_value)
        lin_value = lin(test_grids)
        end_time = time.time()
        print("Spline Approximation:")
        print("Computational Time:", end_time - start_time)
        print("Maximum Absolute Error:", np.max(np.abs(lin_value - real_value)))

        start_time = time.time()
        poly = PolynomialApprox(n)
        poly.approx(even_grids, even_value)
        poly_value = poly(test_grids)
        end_time = time.time()
        print("Polynomial Approximation:")
        print("Computational Time:", end_time - start_time)
        print("Maximum Absolute Error:", np.max(np.abs(poly_value - real_value)))

        start_time = time.time()
        cheby = ChebyshevApprox(
            (self.upper + self.lower) / 2, (self.upper - self.lower) / 2, n
        )
        che_grids = cheby.default_grids
        che_gvalue = self.real_func(che_grids)
        cheby.approx(che_gvalue)
        che_value = cheby(test_grids)
        end_time = time.time()
        print("Chebyshev Approximation:")
        print("Computational Time:", end_time - start_time)
        print("Maximum Absolute Error:", np.max(np.abs(che_value - real_value)))

        if plot:
            plt.plot(test_grids, real_value, label="Real Value")
            plt.plot(test_grids, lin_value, label="Spline Approximation")
            plt.plot(test_grids, poly_value, label="Polynomial Approximation")
            plt.plot(test_grids, che_value, label="Chebyshev Approximation")
            plt.legend()
            plt.title("Approximation of Degree " + str(n))
            plt.show()
