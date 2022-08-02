import gpytorch
from typing import Any
import src.util as util
import torch
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "./data/data.txt"
TRAIN_SIZE = 10
TRAINING_ITER = 1000
MODEL_PATH = "./model.state"


def save_models(optimizer: Any, model: Any, iter: int) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "./iter_{}.state".format(iter),
    )


class GPModel(gpytorch.models.ExactGP):  # type:ignore
    def __init__(
        self,
        train_x: Any,
        train_y: Any,
        likelihood: Any,
    ):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: Any) -> Any:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    train_xs, train_ys, test_xs, test_ys = util.load_dataset(
        DATA_PATH, TRAIN_SIZE
    )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    torch_xs = torch.tensor(train_xs)
    torch_ys = torch.tensor(train_ys)

    model = GPModel(torch_xs, torch_ys, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters()
            },  # Includes GaussianLikelihood parameters
        ],
        lr=0.1,
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(TRAINING_ITER):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(torch_xs)
        # Calc loss and backprop gradients
        loss = -mll(output, torch_ys)
        loss.backward()
        print(
            "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
            % (
                i + 1,
                TRAINING_ITER,
                loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item(),
            )
        )
        optimizer.step()

    save_models(optimizer, model, TRAINING_ITER)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(torch_xs))

    # plt.rcParams["font.size"] = 18
    # plt.figure(figsize=(15,10))
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        mean = observed_pred.mean
        lower = lower.numpy()
        upper = upper.numpy()
        width = upper - lower
        mean = mean.numpy()
        plt.errorbar(train_ys, mean, yerr=width, fmt="o")
        xvalues = np.linspace(0, 400, 100)
        yvalues = np.linspace(0, 400, 100)
        plt.plot(xvalues, yvalues, linestyle="dashed")
        plt.xlabel("ground truth")
        plt.ylabel("prediction")
        plt.legend(loc="best")
        plt.savefig("./result_with_gpytorch.jpg")
