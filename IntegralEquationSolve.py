from CustomNN_Model import CustomClass
from CustomDataSet import CustomDataSet
import torch
import random
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IntegralEquationSolver:
    def __init__(
        self,
        a: float,
        b: float,
        right_part_of_eq: Callable[[float, float], float],
        integral_part: Callable[[float, float, float], float],
        true_solution: Callable[[float], float]
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.left_bound = a
        self.right_bound = b
        self.number_of_points = 20
        self.n_of_integr_points = 100
        self.batch_size = 1

        self.integral_elementary_square = np.abs(
            (self.right_bound - self.left_bound) / (self.n_of_integr_points - 1)
        )
        self.loss = torch.nn.MSELoss()
        n_inputs = 1
        n_hidden_neurons = 200
        n_outputs = 1
        self.nn_model = CustomClass(n_inputs, n_hidden_neurons, n_outputs)
        self.nn_model.to(self.device)
        self.num_epochs = 100

        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=3e-4, momentum = 0.99)
        # self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr = 3e-4, betas=(0.99, 0.9999))
        # self.optimizer = torch.optim.RMSprop(self.nn_model.parameters(), lr = 3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=1
        )

        self.right_part_of_eq = right_part_of_eq

        self.integral_part = integral_part

        self.integral_mid_points = self.get_integral_midpoints()

        self.train_domain, self.valid_domain = self.get_train_valid_domains()

        self.train_dataset, self.valid_dataset = self.get_train_valid_datasets()

        (
            self.train_data_loader,
            self.valid_data_loader,
        ) = self.get_train_valid_dataloaders()


        self.true_solution = true_solution

        self.AbsoluteError = lambda true_solution, approximation: torch.abs(
            true_solution - approximation
        )
        self.MaxAbsoluteError = lambda true_solution, approximation: torch.max(
            self.AbsoluteError(true_solution, approximation)
        )

    def get_1d_domain(self, a, b, n_points):
        domain = torch.linspace(a, b, n_points).unsqueeze(1)
        domain = domain.to(device)
        return domain

    def get_integral_midpoints(self):
        domain = self.get_1d_domain(self.left_bound, self.right_bound, self.n_of_integr_points)
        points = (domain[0:-1] + domain[1:]) / 2
        return points.reshape(self.n_of_integr_points - 1, 1)

    def get_train_valid_domains(self):
        train_domain = self.get_1d_domain(self.left_bound, self.right_bound, self.number_of_points)
        step_aside = np.abs(
            (self.right_bound - self.left_bound) / (2 * self.number_of_points)
        )
        valid_domain = train_domain + step_aside

        return train_domain, valid_domain

    def get_train_valid_datasets(self):
        return CustomDataSet(self.train_domain), CustomDataSet(self.valid_domain)

    def get_train_valid_dataloaders(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )

        valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_dataloader, valid_dataloader

    def plot_function(self, domain, function, title, x_label, y_label):
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both")
        ax.axhline(y=0, color="k")
        ax.axvline(x=0, color="k")
        ax.plot(domain.cpu().detach().numpy(), function.cpu().detach().numpy())
        plt.show()

    def make_report(self):
        self.nn_model.eval()
        analytical_solution_valid = self.true_solution(self.valid_domain)
        analytical_solution_train = self.true_solution(self.train_domain)

        nn_approximation_valid = self.nn_model(
            self.valid_domain.view(self.number_of_points, 1)
        )
        nn_approximation_train = self.nn_model(
            self.train_domain.view(self.number_of_points, 1)
        )

        print(
            "Train max absolute error: {}".format(
                self.MaxAbsoluteError(analytical_solution_train, nn_approximation_train)
            )
        )

        print(
            "Valid max absolute error: {}".format(
                self.MaxAbsoluteError(analytical_solution_valid, nn_approximation_valid)
            )
        )
        self.plot_function(
            self.valid_domain,
            self.AbsoluteError(analytical_solution_valid, nn_approximation_valid),
            "Absolute error on validation domain: true sol - Approximation",
            "X",
            "Error",
        )

        self.plot_function(
            self.valid_domain,
            self.AbsoluteError(analytical_solution_train, nn_approximation_train),
            "Absolute error on train domain: true sol - Approximation",
            "X",
            "Error",
        )

        self.plot_function(
            self.valid_domain, analytical_solution_valid, "True Solution", "domain_value", "Function_value"
        )
        self.plot_function(
            self.valid_domain, nn_approximation_valid, "Approximation", "domain_value", "Function_value"
        )

        epochs = torch.arange(self.num_epochs)

        self.plot_function(
            epochs, self.losses_valid, "MSE loss validation", "epoch", "loss"
        )
        self.plot_function(epochs, self.losses_train, "MSE loss train", "epoch", "loss")

    def epoch_loss_calc(self, data_loader, phase, zero_value):
        running_loss = 0.0
        # Iterate over data.
        for input in data_loader:
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):

                residual = self.right_part_of_eq(input, self.nn_model)

                integral_value = (
                    sum(self.integral_part(input, self.integral_mid_points, self.nn_model))
                    * self.integral_elementary_square
                )

                residual -= integral_value

                loss_value = self.loss(residual, zero_value)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss_value.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss_value.item()
        epoch_loss = running_loss / len(data_loader)
        return epoch_loss

    def train(self):
        self.losses_train = torch.zeros(self.num_epochs)
        self.losses_valid = torch.zeros(self.num_epochs)
        zero_value = torch.zeros(self.batch_size, 1)
        zero_value = zero_value.to(device)
        for epoch in range(self.num_epochs):
            print("Epoch {}/{}:".format(epoch, self.num_epochs - 1), flush=True)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    data_loader = self.train_data_loader
                    self.nn_model.train()  # Set model to training mode
                else:
                    data_loader = self.valid_data_loader
                    self.nn_model.eval()  # Set model to evaluate mode

                epoch_loss = self.epoch_loss_calc(data_loader, phase, zero_value)

                if phase == "train":
                    self.scheduler.step()
                    self.losses_train[epoch] = epoch_loss
                else:
                    self.losses_valid[epoch] = epoch_loss

                print("{} Loss: {:.4f}".format(phase, epoch_loss), flush=True)
