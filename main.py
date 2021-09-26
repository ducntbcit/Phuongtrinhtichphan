# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from IntegralEquationSolve import *


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    left_part_of_eq = lambda x_var, target_func: target_func(x_var) - torch.sin(
        pi * x_var
    )

    true_solution = lambda x_var: torch.sin(pi * x_var) + 2 / pi

    integral_part = lambda x_curr_var, integral_domain_var,  target_func: 0.5 * target_func(integral_domain_var)
    IES = IntegralEquationSolver(0, 1, left_part_of_eq, integral_part, true_solution)
    IES.train()
    IES.make_report()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
