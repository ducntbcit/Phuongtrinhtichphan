import numpy as np


class Definite_integral:
    def __init__(self, integral_function, Gauss_quadratures):
        self.integral_function = integral_function
        self.Gauss_quadratures = Gauss_quadratures

    def calculate_integral(self):
        return (
            (self.Gauss_quadratures.b - self.Gauss_quadratures.a)
            / 2
            * np.sum(
                np.fromiter(
                    (
                        point.weight * self.integral_function(point.global_point)
                        for point in self.Gauss_quadratures.gauss_points
                    ),
                    dtype=float,
                )
            )
        )


# Fredholm's equation type: phi(x) = f(x) + lambda * integral(Kern(x,s)*phi(s)) for s=a to b
class Integral_equation:
    def __init__(self, f, lmd, kernel, left_boundary, right_boundary):
        self.f = f
        self.lmd = lmd
        self.kernel = kernel
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    # def


# f = lambda x: np.sin(x)
