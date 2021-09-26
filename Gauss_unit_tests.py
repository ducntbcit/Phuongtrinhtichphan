import unittest
from gauss import *
import math
from Integral_equation import *

class CheckSTDofApproximation(unittest.TestCase):
    def test_gauss_quadratures_1(self):
        a = 3
        b = 5
        Gauss_t3 = Gauss_quadratures(a,b)
        Gauss_t3.gauss_t3_init()
        correct_answer = [4 , 4 - np.sqrt(0.6), 4 + np.sqrt(0.6)]
        self.assertEqual(correct_answer, Gauss_t3.get_global_points())
    def test_integral_calculation_1(self):
        a = 0
        b = np.pi/2
        Gauss_t20 = Gauss_quadratures(a,b)
        Gauss_t20.gauss_t20_init()
        int_function = lambda x: np.sin(x)
        correct_answer = 1
        acceptable_error = 1e-4
        integral = Definite_integral(int_function, Gauss_t20)
        integral_sum = integral.calculate_integral()
        self.assertTrue(np.abs(correct_answer - integral_sum) < acceptable_error)

    def test_integral_calculation_2(self):
        a = -5
        b = 5
        Gauss_t20 = Gauss_quadratures(a,b)
        Gauss_t20.gauss_t20_init()
        int_function = lambda x: x * math.exp(x)
        correct_answer = 593.69
        acceptable_error = 1e-2
        integral = Definite_integral(int_function, Gauss_t20)
        integral_sum = integral.calculate_integral()
        self.assertTrue(np.abs(correct_answer - integral_sum) < acceptable_error)




if __name__ == '__main__':
    unittest.main()